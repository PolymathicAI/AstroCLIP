import lightning as L
import torch
from torch import nn, optim
from torchvision import models
from torchvision.transforms import (
    Compose,
    GaussianBlur,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)

from astroclip.env import format_with_env

ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")


class SupervisedModel(L.LightningModule):
    def __init__(
        self,
        model_name,
        modality,
        properties,
        scale,
        num_epochs,
        lr=1e-3,
        save_dir=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.modality = modality
        self.properties = properties
        self.scale = scale
        self.lr = lr
        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()
        self.save_dir = save_dir
        self._initialize_model(model_name)
        self.image_transforms = Compose(
            [
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                GaussianBlur(kernel_size=3),
            ]
        )

    def _initialize_model(self, model_name):
        if model_name == "resnet18":
            self.model = ResNet18(n_out=len(self.properties))
        elif model_name == "conv+att":
            self.model = SpectrumEncoder(n_latent=len(self.properties))
        elif model_name == "mlp":
            self.model = MLP(
                n_in=3,
                n_out=len(self.properties),
                n_hidden=(64, 64),
                act=[nn.ReLU()] * 3,
            )
        else:
            raise ValueError("Invalid model name")

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        if self.modality == "image":
            X_batch = self.image_transforms(X_batch)
        y_pred = self(X_batch)
        loss = self.criterion(y_pred, y_batch.squeeze())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        loss = self.criterion(y_pred, y_batch.squeeze())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        return {"optimizer": optimizer, "scheduler": scheduler}


class ResNet18(nn.Module):
    """Modfied ResNet18."""

    def __init__(self, n_out=1):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(512, n_out)

    def forward(self, x):
        return self.resnet(x)


class MLP(nn.Sequential):
    """MLP model"""

    def __init__(self, n_in, n_out, n_hidden=(16, 16, 16), act=None, dropout=0):
        if act is None:
            act = [
                nn.LeakyReLU(),
            ] * (len(n_hidden) + 1)
        assert len(act) == len(n_hidden) + 1

        layer = []
        n_ = [n_in, *n_hidden, n_out]
        for i in range(len(n_) - 2):
            layer.append(nn.Linear(n_[i], n_[i + 1]))
            layer.append(act[i])
            layer.append(nn.Dropout(p=dropout))
        layer.append(nn.Linear(n_[-2], n_[-1]))
        super(MLP, self).__init__(*layer)


class SpectrumEncoder(nn.Module):
    """Spectrum encoder

    Modified version of the encoder by Serrà et al. (2018), which combines a 3 layer CNN
    with a dot-product attention module. This encoder adds a MLP to further compress the
    attended values into a low-dimensional latent space.

    Paper: Serrà et al., https://arxiv.org/abs/1805.03908
    """

    def __init__(self, n_latent, n_hidden=(32, 32), act=None, dropout=0):
        super(SpectrumEncoder, self).__init__()
        self.n_latent = n_latent

        filters = [8, 16, 16, 32]
        sizes = [5, 10, 20, 40]
        self.conv1, self.conv2, self.conv3, self.conv4 = self._conv_blocks(
            filters, sizes, dropout=dropout
        )
        self.n_feature = filters[-1] // 2

        # pools and softmax work for spectra and weights
        self.pool1, self.pool2, self.pool3 = tuple(
            nn.MaxPool1d(s, padding=s // 2) for s in sizes[:3]
        )
        self.softmax = nn.Softmax(dim=-1)

        # small MLP to go from CNN features to latents
        if act is None:
            act = [nn.PReLU(n) for n in n_hidden]
            # last activation identity to have latents centered around 0
            act.append(nn.Identity())
        self.mlp = MLP(
            self.n_feature, self.n_latent, n_hidden=n_hidden, act=act, dropout=dropout
        )

    def _conv_blocks(self, filters, sizes, dropout=0):
        convs = []
        for i in range(len(filters)):
            f_in = 1 if i == 0 else filters[i - 1]
            f = filters[i]
            s = sizes[i]
            p = s // 2
            conv = nn.Conv1d(
                in_channels=f_in,
                out_channels=f,
                kernel_size=s,
                padding=p,
            )
            norm = nn.InstanceNorm1d(f)
            act = nn.PReLU(f)
            drop = nn.Dropout(p=dropout)
            convs.append(nn.Sequential(conv, norm, act, drop))
        return tuple(convs)

    def _downsample(self, x):
        # compression
        x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv4(x)
        C = x.shape[1] // 2
        # split half channels into attention value and key
        h, a = torch.split(x, [C, C], dim=1)

        return h, a

    def forward(self, y):
        # run through CNNs
        h, a = self._downsample(y)
        # softmax attention
        a = self.softmax(a)

        # attach hook to extract backward gradient of a scalar prediction
        # for Grad-FAM (Feature Activation Map)
        if ~self.training and a.requires_grad == True:
            a.register_hook(self._attention_hook)

        # apply attention
        x = torch.sum(h * a, dim=2)

        # run attended features into MLP for final latents
        x = self.mlp(x)
        return x

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _attention_hook(self, grad):
        self._attention_grad = grad

    @property
    def attention_grad(self):
        if hasattr(self, "_attention_grad"):
            return self._attention_grad
        else:
            return None
