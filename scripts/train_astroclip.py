import torch
import numpy as np

from torch.utils.data import DataLoader

from datasets import load_dataset

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from astroclip.networks.spectra import SpectrumEncoder
from astroclip.augmentations import ToRGB, AddGaussianNoise, SpectrumNoising
from astroclip.modules import AstroCLIP

import torchvision.models as models
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, CenterCrop, InterpolationMode

from pl_bolts.models.self_supervised import Moco_v2

class OutputExtractor(L.LightningModule):
    """
    Pass data through network to extract model outputs
    """
    def __init__(
        self,
        backbone: torch.nn.Module,
    ):    
        super(OutputExtractor, self).__init__()

        # pass
        self.backbone = backbone
        self.backbone.train()

    def forward(self, batch):
        
        x, _ = batch
        
        z_emb = self.backbone(x)
       
        return z_emb
    
    def predict(self, batch, batch_idx: int, dataloader_idx: int=None):
        return self(batch)

class ReorderAndShift():
    def __call__(self, tensor):
        tensor = tensor.permute(0, 3, 1, 2)  # Change to [batch_size, nchannel, npix, npix]
        
        # Add random shift by a few pixels
        shift = torch.randint(low=-5, high=5, size=(2,))
        tensor = torch.roll(tensor, shifts=shift.tolist(), dims=(2, 3))
        
        return tensor

def main():
    torch.set_float32_matmul_precision('medium')

    # Instantiate logger
    wandb_logger = WandbLogger(log_model="all", 
                           project='astroclip',
                           name='conv-clip')

    # Load dataset
    dataset = load_dataset('../astroclip/datasets/legacy_survey.py',
                           keep_in_memory=True)
    dataset.set_format(type='torch', columns=['image', 'spectrum'])

    train_loader = DataLoader(dataset['train'], batch_size=1024, 
                          shuffle=True, num_workers=10, pin_memory=True, 
                          drop_last=True)

    val_loader = DataLoader(dataset['test'], batch_size=1024, 
                        shuffle=False, num_workers=10, pin_memory=True, 
                        drop_last=True)

    # Setting up image augmentations 
    image_transforms = Compose([
            ReorderAndShift(),
            AddGaussianNoise(0, 0.03),
            RandomRotation(45, interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            CenterCrop(96),
    ])

    # Loading vector with some information for noising the spectra
    std_spectra = np.load('spectra_std.npz')['spectra_npz'].astype('float32')
    spectrum_augmentations = SpectrumNoising(std_spectra)

    # Build the two model we will be using
    spectrum_encoder = SpectrumEncoder(n_latent=128, instrument=None,
                                       n_hidden=[256,256], dropout=0.1)

    moco_model = Moco_v2.load_from_checkpoint(
                    checkpoint_path='/mnt/home/flanusse/ceph/resnet50.ckpt'
            )
    # extract encoder_q from Moco_v2 model
    backbone = moco_model.encoder_q
    image_encoder = OutputExtractor(backbone).to('cuda')


    model = AstroCLIP(image_encoder, 
                      spectrum_encoder,
                      embedding_dim=128,
                      image_transform=image_transforms,
                      spectrum_augmentation=spectrum_augmentations)

    # Initialize trainer
    trainer = L.Trainer(callbacks=[
                ModelCheckpoint(
                    every_n_epochs=5,
                )],
                logger=wandb_logger
                )
    
    # Train model
    trainer.fit(model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)

if __name__ == '__main__':
    main()
