from torch.utils.data import DataLoader

from datasets import load_dataset

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from astroclip.networks.spectra import SpectrumEncoder
from astroclip.augmentations import ToRGB, AddGaussianNoise
from astroclip.modules import AstroCLIP

import torchvision.models as models
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, CenterCrop, InterpolationMode

def main():
    # Instantiate logger
    wandb_logger = WandbLogger(log_model="all", 
                               project="astroclip",
                               name="clip")

    # Load dataset
    dataset = load_dataset('EiffL/AstroCLIP')
    dataset.set_format(type='torch', columns=['image', 'spectrum'])

    train_loader = DataLoader(dataset['train'], batch_size=2048, 
                          shuffle=True, num_workers=10, pin_memory=True, 
                          drop_last=True)

    val_loader = DataLoader(dataset['test'], batch_size=512, 
                        shuffle=False, num_workers=10, pin_memory=True, 
                        drop_last=True)

    # Setting up data augmentations 
    image_transforms = Compose([
            ToRGB(),
            AddGaussianNoise(0,0.15),
            RandomRotation(45,interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            CenterCrop(96),
    ])

    # Build the two model we will be using
    spectrum_encoder = SpectrumEncoder(n_latent=256, instrument=None)
    image_encoder = models.resnet18(num_classes=256, weights=None)

    model = AstroCLIP(image_encoder, 
                      spectrum_encoder,
                      image_transform=image_transforms)

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
