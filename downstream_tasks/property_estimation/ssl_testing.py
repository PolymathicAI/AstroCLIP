import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from astropy.table import Table, join
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils.helpers import (
    few_shot_train,
    photometry_r2,
    plot_radar,
    resnet_r2,
    spender_r2,
    zero_shot_train,
)
from utils.models import SimpleMLP

from datasets import load_dataset

sns.set()
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})

# Overall Definitions:
properties = ["Z_HP", "LOG_MSTAR", "Z_MW", "t_ageMW", "SFR"]
scaler = StandardScaler()


# ----- Replace with new Dataset Loader ----- #
def get_provabgs(embedding_file, images=True):
    # Retrieve your CLIP embeddings
    CLIP_embeddings = h5py.File(embedding_file, "r")
    train_embeddings = CLIP_embeddings["train"]
    test_embeddings = CLIP_embeddings["test"]

    if images:
        train_table = Table(
            {
                "targetid": train_embeddings["targetid"],
                "image_features": train_embeddings["image_features"],
            }
        )

        test_table = Table(
            {
                "targetid": test_embeddings["targetid"],
                "image_features": test_embeddings["image_features"],
            }
        )

    else:
        train_table = Table(
            {
                "targetid": train_embeddings["targetid"],
                "spectra_features": train_embeddings["spectra_features"],
            }
        )

        test_table = Table(
            {
                "targetid": test_embeddings["targetid"],
                "spectra_features": test_embeddings["spectra_features"],
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


# -------------------------------------------- #


def get_data(embedding_file, images=True):
    train_provabgs, test_provabgs = get_provabgs(embedding_file, images)

    # Scale the galaxy property data
    y_train, y_test = torch.zeros((len(train_provabgs), 5)), torch.zeros(
        (len(test_provabgs), 5)
    )
    for i, p in enumerate(properties):
        prop_train, prop_test = train_provabgs[p].reshape(-1, 1), test_provabgs[
            p
        ].reshape(-1, 1)
        if p == "Z_MW":
            prop_train, prop_test = np.log(prop_train), np.log(prop_test)
        if p == "SFR":
            prop_train, prop_test = np.log(prop_train) - train_provabgs[
                "LOG_MSTAR"
            ].reshape(-1, 1), np.log(prop_test) - test_provabgs["LOG_MSTAR"].reshape(
                -1, 1
            )
        prop_scaler = StandardScaler().fit(prop_train)
        prop_train, prop_test = prop_scaler.transform(
            prop_train
        ), prop_scaler.transform(prop_test)
        y_train[:, i], y_test[:, i] = torch.tensor(
            prop_train.squeeze(), dtype=torch.float32
        ), torch.tensor(prop_test.squeeze(), dtype=torch.float32)

    if images:
        train_images, test_images = (
            train_provabgs["image_features"],
            test_provabgs["image_features"],
        )
        image_scaler = StandardScaler().fit(train_images)
        train_images, test_images = image_scaler.transform(
            train_images
        ), image_scaler.transform(test_images)

        data = {
            "X_train": train_images,
            "X_test": test_images,
            "y_train": y_train,
            "y_test": y_test,
        }

    else:
        train_spectra, test_spectra = (
            train_provabgs["spectra_features"],
            test_provabgs["spectra_features"],
        )
        spectrum_scaler = StandardScaler().fit(train_spectra)
        train_spectra, test_spectra = spectrum_scaler.transform(
            train_spectra
        ), spectrum_scaler.transform(test_spectra)

        data = {
            "X_train": train_spectra,
            "X_test": test_spectra,
            "y_train": y_train,
            "y_test": y_test,
        }

    return data


def main(
    embedding_file,
    save_dir,
    source="images",
    train_type="zero_shot",
    stein_embedding_file="/mnt/home/lparker/ceph/stein_propertyembeddings.h5",
    DINO_embedding_file="/mnt/home/lparker/ceph/DINO_embeddings.h5",
    GalFormer_embedding_file="/mnt/home/lparker/ceph/GalFormer_embeddings.h5",
):
    # Create Directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get Embeddings
    if source == "images":
        data = get_data(embedding_file, images=True)
        stein_data = get_data(stein_embedding_file, images=True)
        DINO_data = get_data(DINO_embedding_file, images=True)

        r2_scores = {}

        # Do Zero Shot with KNN
        if train_type == "zero_shot":
            r2_scores["CLIP"] = zero_shot_train(
                data["X_train"],
                data["y_train"],
                data["X_test"],
                data["y_test"],
                properties,
            )
            r2_scores["Stein et. al."] = zero_shot_train(
                stein_data["X_train"],
                stein_data["y_train"],
                stein_data["X_test"],
                stein_data["y_test"],
                properties,
            )
            r2_scores["DINO"] = zero_shot_train(
                DINO_data["X_train"],
                DINO_data["y_train"],
                DINO_data["X_test"],
                DINO_data["y_test"],
                properties,
            )

        # Do Few-Shot with MLP
        elif train_type == "few_shot":
            r2_scores["CLIP"] = few_shot_train(
                data["X_train"],
                data["y_train"],
                data["X_test"],
                data["y_test"],
                properties,
            )
            r2_scores["Stein et. al."] = few_shot_train(
                stein_data["X_train"],
                stein_data["y_train"],
                stein_data["X_test"],
                stein_data["y_test"],
                properties,
            )
            r2_scores["DINO"] = few_shot_train(
                DINO_data["X_train"],
                DINO_data["y_train"],
                DINO_data["X_test"],
                DINO_data["y_test"],
                properties,
            )

        else:
            raise ValueError("Only train types are zero_shot or few_shot")

        # Get Baselines from ResNet18
        resnet18 = torch.load(
            "./baseline_models/resnet_results", map_location=torch.device(device)
        )
        r2_scores["ResNet18"] = resnet_r2(resnet18, properties)

        # Get Baselines from Photometry
        photo_mlp = torch.load(
            "./baseline_models/photometry_results", map_location=torch.device(device)
        )
        r2_scores["Photometry MLP"] = photometry_r2(photo_mlp)

        # Get Correct Labeling Information
        r2_scores["labels"] = ["Z_{HP}", "LOG_{M_*}", "Z_{MW}", "t_{age}", "SFR"]

        # Plot Images
        save_file = "images_" + train_type
        plot_radar(r2_scores, os.path.join(save_dir, save_file))

        # Print Results
        for key, numbers in r2_scores.items():
            if key != "labels":
                print(f"{key}:")
                for i, num in enumerate(numbers):
                    label = r2_scores["labels"][i]
                    print(f"    {label}: {num}")

    elif source == "spectra":
        data = get_data(embedding_file, images=False)
        GalFormer_data = get_data(GalFormer_embedding_file, images=False)

        r2_scores = {}

        # Do Zero Shot with KNN
        if train_type == "zero_shot":
            r2_scores["CLIP"] = zero_shot_train(
                data["X_train"],
                data["y_train"],
                data["X_test"],
                data["y_test"],
                properties,
            )
            r2_scores["Transformer"] = zero_shot_train(
                GalFormer_data["X_train"],
                GalFormer_data["y_train"],
                GalFormer_data["X_test"],
                GalFormer_data["y_test"],
                properties,
            )

        if train_type == "few_shot":
            r2_scores["CLIP"] = few_shot_train(
                data["X_train"],
                data["y_train"],
                data["X_test"],
                data["y_test"],
                properties,
            )
            r2_scores["Transformer"] = few_shot_train(
                GalFormer_data["X_train"],
                GalFormer_data["y_train"],
                GalFormer_data["X_test"],
                GalFormer_data["y_test"],
                properties,
            )

        # Get Baseline from Spender
        spender = torch.load(
            "./baseline_models/spender_results", map_location=torch.device(device)
        )
        r2_scores["Supervised Spectrum"] = spender_r2(spender, properties)

        # Get Correct Labeling Information
        r2_scores["labels"] = ["Z_{HP}", "LOG_{M_*}", "Z_{MW}", "t_{age}", "SFR"]

        # Plot Images
        save_file = "spectra_" + train_type
        plot_radar(r2_scores, os.path.join(save_dir, save_file))

        # Print Results
        for key, numbers in r2_scores.items():
            if key != "labels":
                print(f"{key}:", flush=True)
                for i, num in enumerate(numbers):
                    label = r2_scores["labels"][i]
                    print(f"    {label}: {num}", flush=True)

    else:
        raise ValueError("Only accepting source of images or spectra")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero- or few- shot learning on images or spectra embeddings"
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        default="/mnt/home/lparker/ceph/newest_embeddings.h5",
        help="Path to the embedding file.",
    )
    parser.add_argument("--save_dir", type=str, help="File to save R2 radar graphs.")
    parser.add_argument(
        "--source", type=str, default="images", help="list spectra or images"
    )
    parser.add_argument(
        "--train_type", type=str, default="zero_shot", help="list zero_shot or few_shot"
    )
    args = parser.parse_args()
    main(
        args.embedding_file,
        args.save_dir,
        source=args.source,
        train_type=args.train_type,
    )
