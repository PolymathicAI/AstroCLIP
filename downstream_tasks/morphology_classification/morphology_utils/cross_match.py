import argparse
import os
from typing import List

import h5py
import numpy as np
import requests
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm

from astroclip.astrodino.data.augmentations import ToRGB
from astroclip.env import format_with_env

gz_5_link = (
    "https://zenodo.org/records/4573248/files/gz_decals_volunteers_5.csv?download=1"
)


def _generate_catalog(files: List[str]) -> Table:
    """Generate a catalog from a list of files."""
    ra_list, dec_list = [], []
    index_list, file_list = [], []
    print("Generating catalogs", flush=True)
    for i, file in enumerate(tqdm(files)):
        with h5py.File(file, "r") as f:
            ra = f["ra"][:]
            dec = f["dec"][:]

            # Append data to lists
            ra_list.extend(ra)
            dec_list.extend(dec)
            file_list.extend([file] * len(ra))
            index_list.extend(range(0, len(ra)))

    # Create astropy table
    return Table(
        [ra_list, dec_list, index_list, file_list], names=("ra", "dec", "index", "file")
    )


def _cross_match_tables(
    table1: Table, table2: Table, max_sep: float = 0.5
) -> tuple[Table, Table]:
    """Cross-match two tables."""

    # Create SkyCoord objects
    coords1 = SkyCoord(ra=table1["ra"] * u.degree, dec=table1["dec"] * u.degree)
    coords2 = SkyCoord(ra=table2["ra"] * u.degree, dec=table2["dec"] * u.degree)

    print("Matching coordinates", flush=True)

    # Match coordinates
    idx, d2d, _ = coords1.match_to_catalog_sky(coords2)

    # Define separation constraint and apply it
    max_sep = max_sep * u.arcsec
    sep_constraint = d2d < max_sep

    print(f"Total number of matches: {np.sum(sep_constraint)} \n", flush=True)
    return table1[sep_constraint], table2[idx[sep_constraint]]


def _get_images(files: list[str], classifications: Table) -> Table:
    """Get images from files."""

    # Set up transforms
    transform = Compose([CenterCrop(144), ToRGB()])

    # Add images to catalog
    print("Adding images to catalog", flush=True)
    images = np.zeros((len(classifications), 3, 144, 144))
    for idx, file in enumerate(files):
        print(f"Processing file: {idx}", flush=True)
        with h5py.File(file, "r") as f:
            for k, entry in tqdm(enumerate(classifications)):
                if entry["file"] != file:
                    continue
                index = entry["index"]
                image = transform(torch.tensor(f["images"][index])).T
                images[k] = np.array(image)
    classifications["image"] = images
    return classifications


def _download_gz5_decals(survey_path: str) -> None:
    """Download Galaxy Zoo 5 classifications."""
    response = requests.get(gz_5_link)
    with open(survey_path, "wb") as f:
        f.write(response.content)


def _get_file_location(root_dir: List[str]) -> List[str]:
    """Get the locations of the Legacy Survey image files."""
    north_path = os.path.join(root_dir, "north")
    south_path = os.path.join(root_dir, "south")

    files_north = [
        os.path.join(
            north_path,
            "images_npix152_0%02d000000_0%02d000000.h5" % (i, i + 1),
        )
        for i in range(14)
    ]
    files_south = [
        os.path.join(
            south_path,
            "images_npix152_0%02d000000_0%02d000000.h5" % (i, i + 1),
        )
        for i in range(62)
    ]

    files = files_north + files_south
    return files


def main(root_dir: str, survey_path: str) -> None:
    """
    Pairs Galaxy Zoo classifications with DECaLS images in an Astropy table.

    Args:
        root_dir (str): Root directory of DECaLS images.
        survey_path (str): Path to Galaxy Zoo survey.

    Returns:
        Table: Table of paired classifications.
    """

    # Get file locations
    files = _get_file_location(root_dir)

    # Load morphology classifications
    if not os.path.exists(survey_path):
        _download_gz5_decals(survey_path)
    morphologies = Table.read(survey_path, format="ascii")

    # Generate catalog of ra, dec, index, file from files
    positions = _generate_catalog(files)

    # Cross-match positions with morphology classifications
    classifications, positions_matched = _cross_match_tables(morphologies, positions)

    # Update classifications with index and file
    classifications["index"] = np.array(positions_matched["index"])
    classifications["file"] = np.array(positions_matched["file"])

    # Get images and add them to classifications
    classifications = _get_images(files, classifications)

    # Save classifications
    save_path = survey_path.replace(".csv", ".h5")
    print(f"Saving paired classifications to {save_path}", flush=True)
    classifications.write(save_path, overwrite=True, format="hdf5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="{ASTROCLIP_ROOT}/datasets/decals",
        help="Root directory of DECaLS images.",
    )
    parser.add_argument(
        "--survey_path",
        type=str,
        default="{ASTROCLIP_ROOT}/datasets/galaxy_zoo/gz5_decals_crossmatched.csv",
        help="Path to Galaxy Zoo survey.",
    )

    args = parser.parse_args()
    main(args.root_dir, args.survey_path)
