import os
from argparse import ArgumentParser

import numpy as np
from astropy.table import Table, join
from datasets import load_from_disk
from provabgs import models as Models
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm

from astroclip.data.datamodule import AstroClipCollator, AstroClipDataloader

provabgs_file = "https://data.desi.lbl.gov/public/edr/vac/edr/provabgs/v1.0/BGS_ANY_full.provabgs.sv3.v0.hdf5"


def _download_data(save_path: str):
    """Download the PROVABGS data from the web and save it to the specified directory."""
    # Check if the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Download the PROVABGS file
    local_path = os.path.join(save_path, "BGS_ANY_full.provabgs.sv3.v0.hdf5")
    if not os.path.exists(local_path):
        print("Downloading PROVABGS data...")
        os.system(f"wget {provabgs_file} -O {local_path}")
        print("Downloaded PROVABGS data successfully!")
    else:
        print("PROVABGS data already exists!")


def _get_best_fit(provabgs: Table):
    """Get the best fit model for each galaxy."""
    m_nmf = Models.NMF(burst=True, emulator=True)

    # Filter out galaxies with no best fit model
    provabgs = provabgs[
        (provabgs["PROVABGS_LOGMSTAR_BF"] > 0)
        * (provabgs["MAG_G"] > 0)
        * (provabgs["MAG_R"] > 0)
        * (provabgs["MAG_Z"] > 0)
    ]

    # Get the thetas and redshifts for each galaxy
    thetas = provabgs["PROVABGS_THETA_BF"][:, :12]
    zreds = provabgs["Z_HP"]

    Z_mw = []  # Stellar Metallicitiy
    tage_mw = []  # Age
    avg_sfr = []  # Star-Forming Region

    print("Calculating best-fit properties using the PROVABGS model...")
    for i in tqdm(range(len(thetas))):
        theta = thetas[i]
        zred = zreds[i]

        # Calculate properties using the PROVABGS model
        Z_mw.append(m_nmf.Z_MW(theta, zred=zred))
        tage_mw.append(m_nmf.tage_MW(theta, zred=zred))
        avg_sfr.append(m_nmf.avgSFR(theta, zred=zred))

    # Add the properties to the table
    provabgs["Z_MW"] = np.array(Z_mw)
    provabgs["TAGE_MW"] = np.array(tage_mw)
    provabgs["AVG_SFR"] = np.array(avg_sfr)
    return provabgs


def main(
    astroclip_path: str,
    provabgs_path: str,
    save_path: str = None,
    batch_size: int = 128,
    num_workers: int = 20,
):
    """Cross-match the AstroCLIP and PROVABGS datasets."""

    # Download the PROVABGS data if it doesn't exist
    if not os.path.exists(provabgs_path):
        _download_data(provabgs_path)

    # Load the AstroCLIP dataset
    dataloader = AstroClipDataloader(
        astroclip_path,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=AstroClipCollator(),
        columns=["image", "targetid", "spectrum"],
    )
    dataloader.setup("fit")

    # Process the images
    train_images, train_spectra, train_targetids = [], [], []
    for batch in tqdm(dataloader.train_dataloader(), desc="Processing train images"):
        train_images.append(batch["image"])
        train_spectra.append(batch["spectrum"])
        train_targetids.append(batch["targetid"])

    test_images, test_spectra, test_targetids = [], [], []
    for batch in tqdm(dataloader.val_dataloader(), desc="Processing test images"):
        test_images.append(batch["image"])
        test_spectra.append(batch["spectrum"])
        test_targetids.append(batch["targetid"])

    print(f"Shape of images is {np.concatenate(train_images).shape[1:]}", flush=True)

    # Create tables for the train and test datasets
    train_table = Table(
        {
            "targetid": np.concatenate(train_targetids),
            "image": np.concatenate(train_images),
            "spectrum": np.concatenate(train_spectra),
        }
    )
    test_table = Table(
        {
            "targetid": np.concatenate(test_targetids),
            "image": np.concatenate(test_images),
            "spectrum": np.concatenate(test_spectra),
        }
    )

    # Load the PROVABGS dataset
    provabgs = Table.read(provabgs_path)

    # Filter out galaxies with no best fit model
    provabgs = provabgs[
        (provabgs["PROVABGS_LOGMSTAR_BF"] > 0)
        * (provabgs["MAG_G"] > 0)
        * (provabgs["MAG_R"] > 0)
        * (provabgs["MAG_Z"] > 0)
    ]

    # Get the best fit model for each galaxy
    provabgs = _get_best_fit(provabgs)

    # Scale the properties
    provabgs["LOG_MSTAR"] = np.log(provabgs["PROVABGS_LOGMSTAR_BF"].data)
    provabgs["sSFR"] = np.log(provabgs["AVG_SFR"].data) - np.log(provabgs["Z_MW"].data)

    # Join the PROVABGS and AstroCLIP datasets
    train_provabgs = join(
        provabgs, train_table, keys_left="TARGETID", keys_right="targetid"
    )
    test_provabgs = join(
        provabgs, test_table, keys_left="TARGETID", keys_right="targetid"
    )
    print("Number of galaxies in train:", len(train_provabgs))
    print("Number of galaxies in test:", len(test_provabgs))

    # Save the paired datasets
    if save_path is None:
        train_provabgs.write(
            provabgs_path.replace("provabgs.hdf5", "provabgs_paired_train.hdf5"),
            overwrite=True,
        )
        test_provabgs.write(
            provabgs_path.replace("provabgs.hdf5", "provabgs_paired_test.hdf5"),
            overwrite=True,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--astroclip_path",
        type=str,
        default="/mnt/ceph/users/polymathic/astroclip/datasets/astroclip_file/",
        help="Path to the AstroCLIP dataset.",
    )
    parser.add_argument(
        "--provabgs_path",
        type=str,
        default="/mnt/ceph/users/polymathic/astroclip/datasets/provabgs/provabgs.hdf5",
        help="Path to the PROVABGS dataset.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the paired datasets.",
    )

    args = parser.parse_args()
    main(
        astroclip_path=args.astroclip_path,
        provabgs_path=args.provabgs_path,
        save_path=args.save_path,
    )
