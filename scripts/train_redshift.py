from torch.utils.data import DataLoader

from datasets import load_dataset

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from astroclip.modules import SpecralRegressor


def main():
    # Instantiate logger
    wandb_logger = WandbLogger(log_model="all", 
                               project="astroclip",
                               name="spectrum2redshift")

    # Load dataset
    dataset = load_dataset('../astroclip/datasets/legacy_survey.py')
    dataset.set_format(type='torch', columns=['spectrum', 'redshift'])

    train_loader = DataLoader(dataset['train'], batch_size=64, 
                          shuffle=True, num_workers=10, pin_memory=True, 
                          drop_last=True)

    val_loader = DataLoader(dataset['test'], batch_size=64, 
                        shuffle=False, num_workers=10, pin_memory=True, 
                        drop_last=True)
    
    # Initialize model to predict redshift from spectra
    model = SpecralRegressor(num_features=1)

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
