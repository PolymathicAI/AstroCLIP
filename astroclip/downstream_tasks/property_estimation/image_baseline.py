import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
from astropy.table import Table, join
from datasets import load_dataset
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import (
    Compose,
    GaussianBlur,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ToPILImage,
    ToTensor,
)
from utils.helpers import resnet_augmentations
from utils.models import ResNet18

# Constants
DEFAULT_CACHE_DIR = "/mnt/ceph/users/lparker/datasets_astroclip"
DEFAULT_PROVABGS = "/mnt/home/lparker/ceph/BGS_ANY_full.provabgs.sv3.v0.hdf5"

# Additional
scaler = StandardScaler()
properties = ["Z_HP", "LOG_MSTAR", "Z_MW", "t_ageMW", "SFR"]


def get_provabgs(provabgs_file, cache_dir):
    dataset = load_dataset(
        "/mnt/home/lparker/Documents/AstroFoundationModel/AstroBaselines/datasets/legacy_survey.py",
        cache_dir=cache_dir,
    )
    dataset.set_format(type="torch", columns=["image", "redshift", "targetid"])

    train_table = Table(
        {
            "targetid": np.array(dataset["train"]["targetid"]),
            "image": np.array(dataset["train"]["image"]),
        }
    )
    test_table = Table(
        {
            "targetid": np.array(dataset["test"]["targetid"]),
            "image": np.array(dataset["test"]["image"]),
        }
    )

    provabgs = Table.read("/mnt/home/lparker/ceph/BGS_ANY_full.provabgs.sv3.v0.hdf5")
    provabgs = provabgs[
        (provabgs["LOG_MSTAR"] > 0)
        * (provabgs["MAG_G"] > 0)
        * (provabgs["MAG_R"] > 0)
        * (provabgs["MAG_Z"] > 0)
    ]
    inds = np.random.permutation(len(provabgs))
    provabgs = provabgs[inds]

    train_provabgs = join(
        provabgs, train_table, keys_left="TARGETID", keys_right="targetid"
    )
    test_provabgs = join(
        provabgs, test_table, keys_left="TARGETID", keys_right="targetid"
    )

    return train_provabgs, test_provabgs


def get_dataloaders(provabgs_file, cache_dir, batch_size=512):
    train_dataset, test_dataset = get_provabgs(provabgs_file, cache_dir)

    X_train, X_test = torch.tensor(
        train_dataset["image"], dtype=torch.float32
    ), torch.tensor(test_dataset["image"], dtype=torch.float32)
    train_mean, train_std = torch.mean(X_train, axis=0), torch.std(X_train, axis=0)
    X_train, X_test = (X_train - train_mean) / train_std, (
        X_test - train_mean
    ) / train_std
    print(train_mean, train_std)

    prop_scalers = {}
    y_train, y_test = torch.zeros((len(X_train), 5)), torch.zeros((len(X_test), 5))
    for i, p in enumerate(properties):
        prop_train, prop_test = train_dataset[p].reshape(-1, 1), test_dataset[
            p
        ].reshape(-1, 1)
        if p == "Z_MW":
            prop_train, prop_test = np.log(prop_train), np.log(prop_test)
        if p == "SFR":
            prop_train, prop_test = np.log(prop_train) - train_dataset[
                "LOG_MSTAR"
            ].reshape(-1, 1), np.log(prop_test) - test_dataset["LOG_MSTAR"].reshape(
                -1, 1
            )
        prop_scaler = StandardScaler().fit(prop_train)
        prop_train, prop_test = prop_scaler.transform(
            prop_train
        ), prop_scaler.transform(prop_test)
        y_train[:, i], y_test[:, i] = torch.tensor(
            prop_train.squeeze(), dtype=torch.float32
        ), torch.tensor(prop_test.squeeze(), dtype=torch.float32)
        prop_scalers[p] = prop_scaler

    total_size = len(X_train)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(
        TensorDataset(X_train, y_train), [train_size, val_size]
    )
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, prop_scalers


def train_CNN(
    model,
    train_loader,
    val_loader,
    test_loader,
    prop_scalers,
    device="cuda",
    num_epochs=50,
):
    # Initialize the model
    model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    best_val_loss = float("inf")

    epochs = tqdm.trange(num_epochs, desc="Training ResNet: ", leave=True)

    # Training loop
    for epoch in epochs:
        train_loss = 0
        model.train()
        for X_batch, y_batch in train_loader:
            images = resnet_augmentations(X_batch.permute(0, 3, 2, 1)).to(device)
            y_pred = model(images).squeeze()
            loss = criterion(y_pred, y_batch.to(device))
            train_loss += loss.item()

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_pred, val_true, val_loss = [], [], 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                images = X_batch.permute(0, 3, 2, 1).to(device)
                y_pred = model(images).squeeze().detach().cpu()
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                val_pred.append(y_pred)
                val_true.append(y_batch)

        val_pred = torch.cat(val_pred).numpy()
        val_true = torch.cat(val_true).numpy()

        val_r2s = {}
        for i, prop in enumerate(properties):
            val_r2s[prop] = r2_score(val_true[:, i], val_pred[:, i])

        if val_loss / len(val_loader) < best_val_loss:
            best_model = model.state_dict()
            best_val_loss = val_loss / len(val_loader)

        # Early stopping
        if epoch > 10 and val_loss / len(val_loader) > 1.5 * best_val_loss:
            break

        epochs.set_description(
            "epoch: {}, train loss: {:.4f}, val loss: {:.4f}, z_hp: {:.4f}".format(
                epoch + 1,
                train_loss / len(train_loader),
                val_loss / len(val_loader),
                val_r2s["Z_HP"],
            )
        )
        epochs.update(1)

    print("Done training...")

    # Evaluation
    print(best_val_loss)
    model.load_state_dict(best_model)

    # Train + Validation Set
    train_pred, train_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            images = X_batch.permute(0, 3, 2, 1).to(device)
            y_pred = model(images).squeeze().detach().cpu()
            train_pred.append(y_pred)
            train_true.append(y_batch)

        for X_batch, y_batch in val_loader:
            images = X_batch.permute(0, 3, 2, 1).to(device)
            y_pred = model(images).squeeze().detach().cpu()
            train_pred.append(y_pred)
            train_true.append(y_batch)

    train_pred = torch.cat(train_pred).numpy()
    train_true = torch.cat(train_true).numpy()

    # Test Set
    test_pred, test_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            images = X_batch.permute(0, 3, 2, 1).to(device)
            y_pred = model(images).squeeze().detach().cpu()
            test_pred.append(y_pred)
            test_true.append(y_batch)

    test_pred = torch.cat(test_pred).numpy()
    test_true = torch.cat(test_true).numpy()

    test_r2s, scalers = {}, {}
    for i, prop in enumerate(properties):
        test_r2s[prop] = r2_score(test_true[:, i], test_pred[:, i])
        scalers[prop] = prop_scalers[prop]

    print(f"R2 Scores on Test Set: {test_r2s}")

    return test_r2s, best_model, train_pred, train_true, test_pred, test_true, scalers


def main(provabgs_file, cache_dir, save_path):
    # Initialize ResNet18 Model
    model = ResNet18(num_classes=5)

    # Get Data
    train_loader, val_loader, test_loader, prop_scalers = get_dataloaders(
        provabgs_file, cache_dir
    )
    print("Training Data Retrieved")

    # Train and Evaluate Model
    (
        test_r2s,
        model,
        train_preds,
        train_trues,
        test_preds,
        test_trues,
        scalers,
    ) = train_CNN(model, train_loader, val_loader, test_loader, prop_scalers)

    # Save Results
    resnet_results = {
        "r2": test_r2s,
        "model": model,
        "test_trues": test_trues,
        "test_preds": test_preds,
        "train_trues": train_trues,
        "train_preds": train_preds,
        "scalers": scalers,
    }
    torch.save(resnet_results, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CNN Baseline for Galaxy Property Estimation"
    )
    parser.add_argument(
        "--provabgs_file",
        type=str,
        default=DEFAULT_PROVABGS,
        help="PROVABGS file location",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Directory of Dataset Script",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./baseline_models/resnet_results",
        help="Path to Saved ResNet",
    )
    args = parser.parse_args()

    # Run
    main(args.provabgs_file, args.cache_dir, args.save_path)
