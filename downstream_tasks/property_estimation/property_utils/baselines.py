import os
from argparse import ArgumentParser
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from astropy.table import Table
from sklearn.metrics import r2_score
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import (
    Compose,
    GaussianBlur,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)
from tqdm import trange

from astroclip.env import format_with_env
from astroclip.models.astroclip import ImageHead
from models import MLP, ResNet18, SpectrumEncoder

# Define transforms for image model
image_transforms = Compose(
    [RandomHorizontalFlip(), RandomVerticalFlip(), GaussianBlur(3)]
)


def setup_supervised_data(
    train_data: Table,
    test_data: Table,
    modality: str,
    properties: List[str],
    batch_size: int = 128,
    train_size: float = 0.8,
):
    """Helper function to set up supervised data for training and testing."""
    # Set up the training data
    if modality == "image":
        X_train, X_test = torch.tensor(
            train_data[modality], dtype=torch.float32
        ), torch.tensor(test_data[modality], dtype=torch.float32)

    elif modality == "spectrum":
        X_train, X_test = torch.tensor(
            train_data[modality], dtype=torch.float32
        ), torch.tensor(test_data[modality], dtype=torch.float32)
        X_train = X_train.squeeze().squeeze()
        X_test = X_test.squeeze().squeeze()

    elif modality == "photometry":
        X_train = torch.tensor(
            np.stack([train_data["MAG_G"], train_data["MAG_R"], train_data["MAG_Z"]]),
            dtype=torch.float32,
        ).permute(1, 0)
        X_test = torch.tensor(
            np.stack([test_data["MAG_G"], test_data["MAG_R"], test_data["MAG_Z"]]),
            dtype=torch.float32,
        ).permute(1, 0)

    # Scale the data
    X_mean, X_std = X_train.mean(), X_train.std()
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    # Set up the property data
    property_data, scale = {}, {}
    for p in properties:
        data = torch.tensor(train_data[p].data, dtype=torch.float32)
        mean, std = data.mean(), data.std()
        property_data[p] = ((data - mean) / std).squeeze()
        scale[p] = {"mean": mean.numpy(), "std": std.numpy()}
    y_train = torch.stack([property_data[p] for p in properties], dim=1)

    # Split the data into training, validation, and test sets
    total_size = len(X_train)
    train_size = int(train_size * total_size)
    train_dataset, val_dataset = random_split(
        TensorDataset(X_train, y_train), [train_size, total_size - train_size]
    )

    # Set up the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, scale


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    modality: str,
    scale: Dict[str, Dict[str, float]],
    properties: List[str],
    device="cuda",
    num_epochs=50,
    lr=1e-3,
):
    """Helper function to train a model."""
    model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    epochs = trange(num_epochs, desc="Training Model: ", leave=True)

    # Training loop
    for epoch in epochs:
        train_loss = 0
        model.train()
        for X_batch, y_batch in train_loader:
            if modality == "image":
                X_batch = image_transforms(X_batch)
            y_pred = model(X_batch.to(device)).squeeze()
            loss = criterion(y_pred, y_batch.to(device).squeeze())
            train_loss += loss.item()

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_pred, val_true, val_loss = [], [], 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch.to(device)).squeeze().detach().cpu()
                loss = criterion(y_pred, y_batch.squeeze())
                val_loss += loss.item()
                val_pred.append(y_pred)
                val_true.append(y_batch)

        val_pred = torch.cat(val_pred).numpy()
        val_true = torch.cat(val_true).numpy()

        # Save the best model
        if val_loss / len(val_loader) < best_val_loss:
            best_model = model.state_dict()
            best_val_loss = val_loss / len(val_loader)

        # Set epoch description
        epochs.set_description(
            "epoch: {}, train loss: {:.4f}, val loss: {:.4f}".format(
                epoch + 1,
                train_loss / len(train_loader),
                val_loss / len(val_loader),
            )
        )
        epochs.update(1)

    return best_model


def get_predictions(model, test_loader, test_provabgs, scale, device="cuda"):
    """Use model to get predictions"""
    test_pred = []
    with torch.no_grad():
        for X_batch in test_loader:
            y_pred = model(X_batch.to(device)).squeeze().detach().cpu()
            test_pred.append(y_pred)
    test_pred = torch.cat(test_pred).numpy()

    pred_dict = {}
    for i, p in enumerate(scale.keys()):
        if len(test_pred.shape) > 1:
            pred_dict[p] = (test_pred[:, i] * scale[p]["std"]) + scale[p]["mean"]
        else:
            pred_dict[p] = (test_pred * scale[p]["std"]) + scale[p]["mean"]
        print(f"{p} R^2: {r2_score(test_provabgs[p], pred_dict[p])}")

    return pred_dict


def calculate_baselines(
    train_dataset: str,
    test_dataset: str,
    save_path: str,
    modality: str,
    model_type: str,
    num_epochs: int = 100,
    learning_rate: float = 5e-4,
    properties: str = None,
    device: str = "cuda",
):
    # Define output directory avoiding collisions
    save_dir_base = os.path.join(save_path, modality, model_type, properties)
    save_dir = save_dir_base
    v_int = 0  # Suffix to add in case of collisions
    while os.path.exists(save_dir):
        print(f"Directory {save_dir} already exists, adding suffix")
        v_int += 1
        save_dir = f"{save_dir_base}-v{v_int}"

    if properties == "redshift":
        property_list = ["Z_HP"]
    elif properties == "global_properties":
        property_list = ["LOG_MSTAR", "Z_MW", "TAGE_MW", "sSFR"]
    else:
        raise ValueError(
            "Invalid properties, choose from redshift or global_properties."
        )

    # Load the data
    train_provabgs = Table.read(train_dataset)
    test_provabgs = Table.read(test_dataset)

    # Get the data loaders & normalization
    train_loader, val_loader, test_loader, scale = setup_supervised_data(
        train_provabgs, test_provabgs, modality, properties=property_list
    )

    # Initialize the model
    if modality == "image":
        if model_type == "ResNet18":
            model = ResNet18(n_out=len(property_list))
        elif model_type == "ViT":
            raise ValueError("Not yet implemented")
        elif model_type == "AstroDINO":
            embed_dim = 1024
            model = nn.Sequential(
                ImageHead(
                    freeze_backbone=False,
                    save_directory=save_dir + "/dino/",
                    embed_dim=embed_dim,
                    model_weights="",
                    config="../../../astroclip/astrodino/config.yaml",
                ),
                nn.Linear(embed_dim, len(property_list)),
            )
        else:
            raise ValueError("Invalid model type")
    elif modality == "spectrum":
        if model_type == "Conv+Att":
            model = SpectrumEncoder(n_latent=len(property_list))
        elif model_type == "ViT":
            raise ValueError("Not yet implemented")
        else:
            raise ValueError("Invalid model type")
    elif modality == "photometry":
        if model_type != "MLP":
            raise ValueError(f"Invalid model value {model_type}")
        model = MLP(
            n_in=3, n_out=len(property_list), n_hidden=[64, 64], act=[nn.ReLU()] * 3
        )

    # Train the model
    best_model = train_model(
        model,
        train_loader,
        val_loader,
        modality,
        scale,
        property_list,
        num_epochs=num_epochs,
        lr=learning_rate,
    )
    model.load_state_dict(best_model)

    # Get the predictions
    pred_dict = get_predictions(model, test_loader, test_provabgs, scale, device=device)

    # Save the model and the predictions

    print(f"Saving in {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model, os.path.join(save_dir, "model.pt"))
    torch.save(pred_dict, os.path.join(save_dir, "test_pred.pt"))


if __name__ == "__main__":
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")
    parser = ArgumentParser()
    parser.add_argument(
        "--train_dataset",
        type=str,
        help="Path to the training dataset",
        default=f"{ASTROCLIP_ROOT}/datasets/provabgs/provabgs_paired_train.hdf5",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        help="Path to the test dataset",
        default=f"{ASTROCLIP_ROOT}/datasets/provabgs/provabgs_paired_test.hdf5",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save the model and predictions",
        default=f"{ASTROCLIP_ROOT}/supervised/",
    )
    parser.add_argument(
        "--modality",
        type=str,
        help="Modality of the data ('image', 'spectrum', 'photometry')",
        default="image",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Model to use (e.g. 'ResNet18', 'AstroDINO', 'ViT', 'Conv+Att', 'MLP' )",
        default=None,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train the model",
        default=50,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate for the optimizer",
        default=5e-4,
    )
    parser.add_argument(
        "--properties",
        type=str,
        help="Properties to predict ('redshift' or 'global_properties')",
        default="global_properties",
    )
    args = parser.parse_args()

    # Infer model_type if missing
    if args.model_type == None:
        if args.modality == "image":
            model_type = "ResNet18"
        elif args.modality == "spectrum":
            model_type = "Conv+Att"
        elif args.modality == "photometry":
            model_type = "MLP"
    else:
        model_type = args.model_type

    calculate_baselines(
        train_dataset=args.train_dataset,
        test_dataset=args.test_dataset,
        save_path=args.save_dir,
        modality=args.modality,
        model_type=model_type,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        properties=args.properties,
    )
