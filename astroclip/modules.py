from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import lightning as L
from astroclip.networks.spectra import SpectraEncoder


class SpecralRegressor(L.LightningModule):
    """ Dedicated regression module from spectra using  an
    mse loss function.
    """
    def __init__(self, num_features=1):
        super().__init__()
        self.backbone = SpectraEncoder(None, 512)
        self.fc = nn.Linear(512, num_features)

    def forward(self, x):
        net = self.backbone(x)
        return self.fc(net)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat.squeeze(), y.squeeze())
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]
