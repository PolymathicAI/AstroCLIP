import lightning as L
import torch
import torch.nn.functional as F
import torchvision
from lightning import Trainer
from numpy import ndarray
from sklearn.neighbors import KNeighborsRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from tqdm import tqdm


def few_shot(
    model: nn.Module,
    X_train: ndarray,
    y_train: ndarray,
    X_test: ndarray,
    max_epochs: int = 10,
    hidden_dims: list[int] = [64, 64],
    lr: float = 1e-3,
) -> ndarray:
    """Train a few-shot model using a simple neural network"""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    num_features = y_train.shape[1] if len(y_train.shape) > 1 else 1
    model = MLP(
        n_in=X_train.shape[1],
        n_out=num_features,
        n_hidden=hidden_dims,
        act=[nn.ReLU()] * (len(hidden_dims) + 1),
        dropout=0.1,
    )

    # Set up the model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    model.cuda()
    model.train()
    for epoch in range(max_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.cuda()).squeeze()
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Make predictions
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32).cuda()).cpu().numpy()
    return preds


def zero_shot(
    X_train: ndarray, y_train: ndarray, X_test: ndarray, n_neighbors: int = 64
) -> ndarray:
    """Train a zero-shot model using KNN"""
    neigh = KNeighborsRegressor(weights="distance", n_neighbors=64)
    neigh.fit(X_train, y_train)
    preds = neigh.predict(X_test)
    return preds


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
