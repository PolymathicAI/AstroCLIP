import os

import numpy as np
from tqdm import tqdm
from astropy.table import Table, join
from datasets import load_from_disk
from argparse import ArgumentParser

from provabgs import models as Models


provabgs_file = 'https://data.desi.lbl.gov/public/edr/vac/edr/provabgs/v1.0/BGS_ANY_full.provabgs.sv3.v0.hdf5'


def _download_data(save_path: str):
    """Download the PROVABGS data from the web and save it to the specified directory."""
    # Check if the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Download the PROVABGS file
    local_path = os.path.join(save_path, 'BGS_ANY_full.provabgs.sv3.v0.hdf5')
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
    provabgs = provabgs[(provabgs['PROVABGS_LOGMSTAR_BF'] > 0) *
                   (provabgs['MAG_G'] > 0) *
                   (provabgs['MAG_R'] > 0) *
                   (provabgs['MAG_Z'] > 0)]

    # Get the thetas and redshifts for each galaxy
    thetas = provabgs['PROVABGS_THETA_BF'][:, :12]
    zreds = provabgs['Z_HP']

    Z_mw    = [] # Stellar Metallicitiy
    tage_mw = [] # Age
    avg_sfr = [] # Star-Forming Region

    print("Calculating best-fit properties using the PROVABGS model...")
    for i in tqdm(range(len(thetas))):
        theta = thetas[i]
        zred = zreds[i]

        # Calculate properties using the PROVABGS model
        Z_mw.append(m_nmf.Z_MW(theta, zred=zred))
        tage_mw.append(m_nmf.tage_MW(theta, zred=zred))
        avg_sfr.append(m_nmf.avgSFR(theta, zred=zred))

    # Add the properties to the table
    provabgs['Z_MW'] = np.array(Z_mw)
    provabgs['TAGE_MW'] = np.array(tage_mw)
    provabgs['AVG_SFR'] = np.array(avg_sfr)
    return provabgs


def main(
    astroclip_path: str,
    provabgs_path: str,
    save_path: str = None,
):
    """Cross-match the AstroCLIP and PROVABGS datasets."""

    # Download the PROVABGS data if it doesn't exist
    if not os.path.exists(provabgs_path):
        _download_data(provabgs_path)

    # Load the AstroCLIP dataset
    dataset = load_from_disk(astroclip_path)
    dataset.set_format(type="torch", columns=["image", "redshift", "targetid"])

    # Load the PROVABGS dataset
    provabgs = Table.read(provabgs_path)

    # Filter out galaxies with no best fit model
    provabgs = provabgs[(provabgs['PROVABGS_LOGMSTAR_BF'] > 0) *
                   (provabgs['MAG_G'] > 0) *
                   (provabgs['MAG_R'] > 0) *
                   (provabgs['MAG_Z'] > 0)]
    
    # Get the best fit model for each galaxy
    provabgs = _get_best_fit(provabgs)

    # Create tables for the train and test datasets
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

    # Join the PROVABGS and AstroCLIP datasets
    train_provabgs = join(
        provabgs, train_table, keys_left="TARGETID", keys_right="targetid"
    )
    test_provabgs = join(
        provabgs, test_table, keys_left="TARGETID", keys_right="targetid"
    )

    # Save the paired datasets
    if save_path is None:
        train_provabgs.write(provabgs_path.replace("provabgs.hdf5", "provabgs_paired_train.hdf5"), overwrite=True)
        test_provabgs.write(provabgs_path.replace("provabgs.hdf5", "provabgs_paired_test.hdf5"), overwrite=True)

if __name__ == '__main__':
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




    