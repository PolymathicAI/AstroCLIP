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
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, CenterCrop, InterpolationMode, ToTensor

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

def main():
    # Instantiate logger
    wandb_logger = WandbLogger(log_model="all", 
                           project='astroclip',
                           name='conv-clip')

    # Load dataset
    dataset = load_dataset('../astroclip/datasets/legacy_survey.py',
                           keep_in_memory=True)
    dataset.set_format(type='torch', columns=['image', 'spectrum'])

    # Setting up image augmentations 
    gpu_transforms = Compose([
            AddGaussianNoise(0,0.15),
    ])
    cpu_augmentations = Compose([
            #ToRGB(),
            RandomRotation(45),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            CenterCrop(96),
    ])
    tt = lambda x: torch.from_numpy(np.array(x).astype('float32'))

    dataset.set_transform(lambda x: {'image': cpu_augmentations(tt(x['image']).transpose(1,3)), 'spectrum': tt(x['spectrum'])})

    train_loader = DataLoader(dataset['train'], batch_size=1024, 
                          shuffle=True, num_workers=10, pin_memory=True, 
                          drop_last=True)

    val_loader = DataLoader(dataset['test'], batch_size=1024, 
                        shuffle=False, num_workers=10, pin_memory=True, 
                        drop_last=True)

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
                      image_transform=gpu_transforms,
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
