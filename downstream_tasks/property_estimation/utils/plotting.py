import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score


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
