import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class SupervisedDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data,
        test_data,
        modality,
        properties,
        batch_size=128,
        train_size=0.8,
    ):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.modality = modality
        self.properties = properties
        self.batch_size = batch_size
        self.train_size = train_size

        if modality not in ["image", "spectrum", "photometry"]:
            raise ValueError("Invalid modality")

    def setup(self, stage=None):
        if self.modality == "image":
            # Load the data
            X_train, X_test = torch.tensor(
                self.train_data[self.modality], dtype=torch.float32
            ), torch.tensor(self.test_data[self.modality], dtype=torch.float32)
        elif self.modality == "spectrum":
            X_train, X_test = torch.tensor(
                self.train_data[self.modality], dtype=torch.float32
            ).squeeze(-1), torch.tensor(
                self.test_data[self.modality], dtype=torch.float32
            ).squeeze(
                -1
            )
        elif self.modality == "photometry":
            # Load the photometry data
            X_train = torch.tensor(
                np.stack(
                    [
                        self.train_data["MAG_G"],
                        self.train_data["MAG_R"],
                        self.train_data["MAG_Z"],
                    ]
                ),
                dtype=torch.float32,
            ).permute(1, 0)
            X_test = torch.tensor(
                np.stack(
                    [
                        self.test_data["MAG_G"],
                        self.test_data["MAG_R"],
                        self.test_data["MAG_Z"],
                    ]
                ),
                dtype=torch.float32,
            ).permute(1, 0)

            # Normalize photometry
            mean, std = X_train.mean(), X_train.std()
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std

        # Set up the property data
        property_data, scale = {}, {}
        for p in self.properties:
            data = torch.tensor(self.train_data[p].data, dtype=torch.float32)
            mean, std = data.mean(), data.std()
            property_data[p] = ((data - mean) / std).squeeze()
            scale[p] = {"mean": mean.numpy(), "std": std.numpy()}
        y_train = torch.stack([property_data[p] for p in self.properties], dim=1)

        # Split the data into training, validation, and test sets
        total_size = len(X_train)
        train_size = int(self.train_size * total_size)
        self.train_dataset, self.val_dataset = random_split(
            TensorDataset(X_train, y_train), [train_size, total_size - train_size]
        )

        self.test_dataset = TensorDataset(X_test)
        self.scale = scale

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
