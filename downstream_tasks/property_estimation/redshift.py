import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from astropy.table import Table, join
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from utils.models import MLP, few_shot, zero_shot


def plot_redshift_scatter(preds, z_test, save_loc="scatter.png"):
    """Functionality to plot redshift scatter plots for different models."""
    fig, ax = plt.subplots(2, len(preds.keys()), figsize=(16, 10))

    for i, name in enumerate(preds.keys()):
        sns.scatterplot(ax=ax[0, i], x=z_test, y=preds[name], s=5, color=".15")
        sns.histplot(
            ax=ax[0, i], x=z_test, y=preds[name], bins=50, pthresh=0.1, cmap="mako"
        )
        sns.kdeplot(
            ax=ax[0, i], x=z_test, y=preds[name], levels=5, color="k", linewidths=1
        )

        ax[0, i].plot(0, 0.65, "--", linewidth=1.5, alpha=0.5, color="grey")
        ax[0, i].set_xlim(0, 0.6)
        ax[0, i].set_ylim(0, 0.6)
        ax[0, i].text(
            0.9,
            0.1,
            "$R^2$ score: %0.2f" % r2_score(z_test, preds[name]),
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=22,
            transform=ax[0, i].transAxes,
        )
        ax[0, i].set_title(name, fontsize=25)

    ax[0, 0].set_ylabel("$Z_{pred}$", fontsize=25)

    for i, name in enumerate(preds.keys()):
        x = z_test
        y = (z_test - preds[name]) / (1 + z_test)

        bins = np.linspace(0, 0.62, 20)
        x_binned = np.digitize(x, bins)
        y_avg = [y[x_binned == i].mean() for i in range(1, len(bins))]
        y_std = [y[x_binned == i].std() for i in range(1, len(bins))]

        sns.scatterplot(ax=ax[1, i], x=x, y=y, s=2, alpha=0.3, color="black")
        sns.lineplot(ax=ax[1, i], x=bins[:-1], y=y_std, color="r", label="std")

        # horizontal line on y = 0
        ax[1, i].axhline(0, color="grey", linewidth=1.5, alpha=0.5, linestyle="--")

        # sns.scatterplot(ax=ax[1,i], x=bins[:-1], y=y_avg, s=15, color='.15')
        ax[1, i].set_xlim(0, 0.6)
        ax[1, i].set_ylim(-0.3, 0.3)
        ax[1, i].set_xlabel("$Z_{true}$", fontsize=25)
        ax[1, i].legend(fontsize=15, loc="upper right")

    ax[1, 0].set_ylabel("$(Z_{true}-Z_{pred})/(1+Z_{true})$", fontsize=25)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_loc, dpi=300)


def main(
    train_embeddings: str,
    test_embeddings: str,
    supervised_model_path: str,
    models: list = ["astroclip", "astrodino", "stein"],
):
    # Get embeddings and PROVABGS table
    train_provabgs = Table.read(train_embeddings)
    test_provabgs = Table.read(test_embeddings)

    # Get data
    data = {}
    for model in models:
        data[model] = {}
        X_train, X_test = (
            train_provabgs[model + "_embeddings"],
            test_provabgs[model + "_embeddings"],
        )
        embedding_scaler = StandardScaler().fit(X_train)
        data[model]["train"] = embedding_scaler.transform(X_train)
        data[model]["test"] = embedding_scaler.transform(X_test)

    # Get redshifts
    z_train = train_provabgs["Z_HP"]
    z_test = test_provabgs["Z_HP"]

    # Scale properties
    scaler = {"mean": z_train.mean(), "std": z_train.std()}
    z_train = (z_train - scaler["mean"]) / scaler["std"]

    # Perfrom knn and mlp
    preds_knn, preds_mlp = {}, {}
    for key in data.keys():
        print(f"Evaluating {key} model...")
        raw_preds_knn = zero_shot(data[key]["train"], z_train, data[key]["test"])
        raw_preds_mlp = few_shot(
            model, data[key]["train"], z_train, data[key]["test"]
        ).squeeze()
        preds_knn[key] = raw_preds_knn * scaler["std"] + scaler["mean"]
        preds_mlp[key] = raw_preds_mlp * scaler["std"] + scaler["mean"]

    # Get predictions from supervised models
    resnet_preds = torch.load(
        os.path.join(supervised_model_path, "image/Z_HP/test_pred.pt")
    )["Z_HP"]
    photometry_preds = torch.load(
        os.path.join(supervised_model_path, "photometry/Z_HP/test_pred.pt")
    )["Z_HP"]

    # Add predictions to dictionary
    preds_knn["resnet18"] = resnet_preds
    preds_knn["photometry"] = photometry_preds
    preds_mlp["resnet18"] = resnet_preds
    preds_mlp["photometry"] = photometry_preds

    # Plot scatter plots
    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")
    plot_redshift_scatter(
        preds_knn, z_test, save_loc="./outputs/redshift_scatter_knn.png"
    )
    plot_redshift_scatter(
        preds_mlp, z_test, save_loc="./outputs/redshift_scatter_mlp.png"
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train_embeddings",
        type=str,
        help="Path to train embeddings",
        default="/mnt/ceph/users/polymathic/astroclip/datasets/provabgs/provabgs_paired_train_embeddings.hdf5",
    )
    parser.add_argument(
        "--test_embeddings",
        type=str,
        help="Path to test embeddings",
        default="/mnt/ceph/users/polymathic/astroclip/datasets/provabgs/provabgs_paired_test_embeddings.hdf5",
    )
    parser.add_argument(
        "--supervised_model_path",
        type=str,
        help="Path to supervised models",
        default="/mnt/ceph/users/polymathic/astroclip/supervised/",
    )
    parser.add_argument(
        "--models",
        type=list,
        default=["astrodino", "stein"],
        help="Models to use for redshift estimation",
    )
    args = parser.parse_args()

    main(
        args.train_embeddings,
        args.test_embeddings,
        args.supervised_model_path,
        args.models,
    )
