from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import lightning as L
import datasets
from astroclip.modules import SpecralRegressor


def main():
    # Load dataset
    dataset = datasets.load('legacy_survey')  # or datasets.LegacySurvey
    train_loader = DataLoader(dataset['train'], batch_size=64, 
                              shuffle=True, num_workers=10, pin_memory=True, 
                              drop_last=True)

    val_loader = DataLoader(dataset['val'], batch_size=64, 
                            shuffle=False, num_workers=10, pin_memory=True, 
                            drop_last=True)

    # Initialize model to predict redshift from spectra
    model = SpecralRegressor(num_features=1)

    # Initialize trainer
    trainer = L.Trainer(gpus=1, max_epochs=10, 
                        callbacks=[ModelCheckpoint(dirpath='checkpoints')])

    # Train model
    trainer.fit(model, 
                train_dataloader=train_loader, 
                val_dataloaders=val_loader)

if __name__ == '__main__':
    main()
