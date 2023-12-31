from torch import nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import lightning as L
import numpy as np
import torch.nn.functional as F

from astroclip.networks.spectra import SpectrumEncoder
from astroclip.losses import CLIPLoss

class SpecralRegressor(L.LightningModule):
    """ Dedicated regression module from spectra using  an
    mse loss function.
    """
    def __init__(self, num_features=1, augmentation=None):
        super().__init__()
        self.backbone = SpectrumEncoder(None, 512)
        self.fc = nn.Linear(512, num_features)
        self.spectrum_augmentation=augmentation

    def forward(self, x):
        net = self.backbone(x)
        return self.fc(net)
    
    @torch.no_grad()
    def on_after_batch_transfer(self, batch, dataloader_idx):
        sp = batch['spectrum']

        if self.trainer.training:
            sp = self.spectrum_augmentation(sp) if self.spectrum_augmentation is not None else sp

        batch['spectrum'] = sp
        return batch

    def training_step(self, batch, batch_idx):
        x = batch['spectrum']
        y = batch['redshift']
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat.squeeze(), y.squeeze())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['spectrum']
        y = batch['redshift']
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat.squeeze(), y.squeeze())
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]


class AstroCLIP(L.LightningModule):

    def __init__(self, image_encoder, 
                       spectrum_encoder,
                       embedding_dim=512,
                       image_augmentation=None,
                       image_transform=None,
                       spectrum_augmentation=None,
                       spectrum_transform=None,
                       loss='CLIP'):
        super().__init__()

        self.image_encoder = image_encoder
        image_last_layer_dim = 128 #list(image_encoder.children())[-1].out_features
        self.image_projection = nn.Linear(image_last_layer_dim, embedding_dim)
        
        self.spectrum_encoder = spectrum_encoder
        # TODO: this is a hacky way to get the last layer of the spectrum encoder
        spectrum_last_layer_dim = 128#list(list(spectrum_encoder.children())[-1].children())[-1].out_features
        self.spectrum_projection = nn.Linear(spectrum_last_layer_dim, embedding_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))#) + 1e-2

        self.image_augmentation = image_augmentation
        self.image_transform = image_transform
        self.spectrum_augmentation = spectrum_augmentation
        self.spectrum_transform = spectrum_transform

        if loss == 'CLIP':
            self.criterion = CLIPLoss()
        else:
            raise NotImplementedError
            
    def forward(self, image=None, spectrum=None):
        image = (image, None)
        if image is not None and spectrum is None:
            embedding = nn.functional.normalize(self.image_projection(self.image_encoder(image)), p=2, dim=-1)
        elif image is None and spectrum is not None:
            embedding = nn.functional.normalize(self.spectrum_projection(self.spectrum_encoder(spectrum)), p=2, dim=-1)
        else:
            raise ValueError("Either image or spectrum must be provided.")
        return embedding
    
    @torch.no_grad()
    def on_after_batch_transfer(self, batch, dataloader_idx):
        im, sp = batch['image'], batch['spectrum']

        if self.trainer.training:
            im = self.image_augmentation(im) if self.image_augmentation is not None else im
            sp = self.spectrum_augmentation(sp) if self.spectrum_augmentation is not None else sp

        im = self.image_transform(im) if self.image_transform is not None else im
        sp = self.spectrum_transform(sp) if self.spectrum_transform is not None else sp

        return {'image': im, 'spectrum': sp}
    
    def training_step(self, batch, batch_idx):
        im, sp = batch['image'], batch['spectrum']
        im = (im, None)
        image_features = nn.functional.normalize(self.image_projection(self.image_encoder(im)), p=2, dim=-1)
        spectrum_features = nn.functional.normalize(self.spectrum_projection(self.spectrum_encoder(sp)), p=2, dim=-1)
        
        loss = self.criterion(image_features, spectrum_features, F.softplus(self.logit_scale))

        self.log("train_loss", loss)
        self.log("scale", F.softplus(self.logit_scale))
        return loss

    def validation_step(self, batch, batch_idx):
        im, sp = batch['image'], batch['spectrum'].squeeze()
        im = (im, None)
        image_features = nn.functional.normalize(self.image_projection(self.image_encoder(im)), p=2, dim=-1)
        spectrum_features = nn.functional.normalize(self.spectrum_projection(self.spectrum_encoder(sp)), p=2, dim=-1)
        
        loss = self.criterion(image_features, spectrum_features, 1.0)

        self.log("val_loss", loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer