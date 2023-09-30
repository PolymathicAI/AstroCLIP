from torch import nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import lightning as L
import numpy as np

from astroclip.networks.spectra import SpectrumEncoder
from astroclip.losses import CLIPLoss

class SpecralRegressor(L.LightningModule):
    """ Dedicated regression module from spectra using  an
    mse loss function.
    """
    def __init__(self, num_features=1):
        super().__init__()
        self.backbone = SpectrumEncoder(None, 512)
        self.fc = nn.Linear(512, num_features)

    def forward(self, x):
        net = self.backbone(x)
        return self.fc(net)

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
        image_last_layer_dim = list(image_encoder.children())[-1].out_features
        self.image_projection = nn.Linear(image_last_layer_dim, embedding_dim)
        
        self.spectrum_encoder = spectrum_encoder
        spectrum_last_layer_dim = list(spectrum_encoder.children())[-1].out_features
        self.spectrum_projection = nn.Linear(spectrum_last_layer_dim, embedding_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).softplus() + 1e-2

        self.image_augmentation = image_augmentation
        self.image_transform = image_transform
        self.spectrum_augmentation = spectrum_augmentation
        self.spectrum_transform = spectrum_transform

        if loss == 'CLIP':
            self.criterion = CLIPLoss()
        else:
            raise NotImplementedError
            
    def forward(self, image=None, spectrum=None):
        if image is not None and spectrum is None:
            embedding = nn.functional.normalize(self.image_projection(self.image_encoder(image)), p=2, dim=-1)
        elif image is None and spectrum is not None:
            embedding = nn.functional.normalize(self.spectrum_projection(self.spectrum_encoder(spectrum)), p=2, dim=-1)
        else:
            raise ValueError("Either image or spectrum must be provided.")
        return embedding
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        im, sp = batch['image'], batch['spectrum']

        if self.trainer.training:
            im = self.image_augmentation(im) if self.image_augmentation is not None else im
            sp = self.spectrum_augmentation(sp) if self.spectrum_augmentation is not None else sp

        im = self.image_transform(im) if self.image_augmentation is not None else im
        sp = self.spectrum_transform(sp) if self.spectrum_augmentation is not None else sp

        return {'image': im, 'spectrum': sp}
    
    def training_step(self, batch, batch_idx):
        im, sp = batch['image'], batch['spectrum']
        
        image_features = self(image=im)
        spectrum_features = self(spectrum=sp)
        
        loss = self.criterion(image_features, spectrum_features, self.logit_scale)

        self.log("train_loss", loss)
        self.log("scale", self.logit_scale)
        return loss

    def validation_step(self, batch, batch_idx):
        im, sp = batch['image'], batch['spectrum'].squeeze()
        
        image_features = self(image=im)
        spectrum_features = self(spectrum=sp)
        
        loss = self.criterion(image_features, spectrum_features, self.logit_scale)

        self.log("val_loss", loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer