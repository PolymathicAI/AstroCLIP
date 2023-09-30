from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from datasets import load_dataset
from astroclip.modules import SpecralRegressor
from pytorch_lightning.loggers import WandbLogger


def main():
    # Instantiate logger
    wandb_logger = WandbLogger(log_model="all", project="astroclip")

    # Load dataset
    dataset = load_dataset('legacy_survey')
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
                train_dataloader=train_loader, 
                val_dataloaders=val_loader)

if __name__ == '__main__':
    main()
