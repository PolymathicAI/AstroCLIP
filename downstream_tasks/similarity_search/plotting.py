import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d


def plot_similar_images(
    query_images: list, sims: dict, similarity_type: str = "im_sim"
):
    """Functionality for plotting retrieved galaxy images"""
    plt.figure(figsize=[19.4, 6.1])
    for n, img in enumerate(query_images):
        plt.subplot(len(query_images), 13, n * 13 + 1)
        plt.imshow(img.T)
        plt.axis("off")
        for j in range(8):
            plt.subplot(len(query_images), 13, n * 13 + j + 1 + 1)
            plt.imshow(sims[n][similarity_type][j].T)
            plt.axis("off")
    plt.subplots_adjust(wspace=0.01, hspace=0.0)
    plt.subplots_adjust(wspace=0.00, hspace=0.01)


def plot_similar_spectra(
    query_spectra: list, query_images: list, sims: dict, similarity_type: str = "im_sim"
):
    """Functionality for plotting retrieved galaxy spectra"""
    l = np.linspace(3586.7408577, 10372.89543574, query_spectra[0].shape[0])
    figure = plt.figure(figsize=[15, 5])
    colors = ["r", "b", "g", "y", "m"]
    for n, sp in enumerate(query_spectra):
        plt.subplot(1, len(query_spectra), n + 1)
        plt.plot(
            l,
            gaussian_filter1d(sp[:, 0], 5),
            color=colors[n],
            lw=1,
            label="spectrum of query image",
        )

        for j in range(5):
            if j == 0:
                plt.plot(
                    l,
                    gaussian_filter1d(sims[n][similarity_type][j + 1][:, 0], 5),
                    alpha=0.5,
                    lw=1,
                    color="gray",
                    label="retrieved spectra",
                )
            else:
                plt.plot(
                    l,
                    gaussian_filter1d(sims[n][similarity_type][j + 1][:, 0], 5),
                    alpha=0.5,
                    lw=1,
                    color="gray",
                )

        plt.xlabel(r"$\lambda$")
        plt.ylabel("flux")
        plt.legend()

        # Add inset image to the first subplot
        axins = plt.gca().inset_axes([0, 0.55, 0.4, 0.4])
        image_data = query_images[n]
        axins.imshow(image_data.T)
        axins.axis("off")
