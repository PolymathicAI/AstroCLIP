import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from astropy.table import Table, join
from datasets import load_dataset
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
from utils.helpers import few_shot_train
from utils.models import SimpleMLP

properties = ["Z_HP", "LOG_MSTAR", "Z_MW", "t_ageMW", "SFR"]
scaler = StandardScaler()


def get_provabgs(provabgs_file, cache_dir):
    dataset = load_dataset("../datasets/legacy_survey.py", cache_dir=cache_dir)
    dataset.set_format(type="torch", columns=["image", "redshift", "targetid"])

    train_table = Table({"targetid": np.array(dataset["train"]["targetid"])})
    test_table = Table({"targetid": np.array(dataset["test"]["targetid"])})

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


def get_data(provabgs_file, cache_dir, batch_size=256):
    train_dataset, test_dataset = get_provabgs(provabgs_file, cache_dir)

    X_train = torch.tensor(
        np.stack(
            [train_dataset["MAG_G"], train_dataset["MAG_R"], train_dataset["MAG_Z"]]
        ),
        dtype=torch.float32,
    ).permute(1, 0)
    X_test = torch.tensor(
        np.stack([test_dataset["MAG_G"], test_dataset["MAG_R"], test_dataset["MAG_Z"]]),
        dtype=torch.float32,
    ).permute(1, 0)

    train_mean, train_std = torch.mean(X_train, axis=0), torch.std(X_train, axis=0)
    X_train, X_test = (X_train - train_mean) / train_std, (
        X_test - train_mean
    ) / train_std

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

    return X_train, y_train, X_test, y_test


def main(provabgs_file, cache_dir, save_file):
    # Get data
    X_train, y_train, X_test, y_test = get_data(provabgs_file, cache_dir)

    # Define Model
    model = SimpleMLP(input_dim=3, output_dim=5)

    # R2 Calculation
    r2s, preds = few_shot_train(
        model, X_train, y_train, X_test, y_test, properties, r2_only=False
    )
    photometry_results = {"r2": r2s, "test_preds": preds, "test_trues": y_test}

    # Save
    torch.save(photometry_results, save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Photometry MLP Baseline for Galaxy Property Estimation"
    )
    parser.add_argument("--save_path", type=str, help="Model output path")
    parser.add_argument(
        "--cache_dir", type=str, default="/mnt/ceph/users/lparker/datasets_astroclip"
    )
    parser.add_argument(
        "--provabgs_file",
        type=str,
        default="/mnt/home/lparker/ceph/BGS_ANY_full.provabgs.sv3.v0.hdf5",
    )
    args = parser.parse_args()

    main(args.provabgs_file, args.cache_dir, args.save_path)
