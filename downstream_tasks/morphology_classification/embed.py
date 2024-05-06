import os
import sys
from argparse import ArgumentParser
from typing import Dict

import torch
from astropy.table import Table
from dinov2.eval.setup import setup_and_build_model
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm

from astroclip.astrodino.data.augmentations import ToRGB


def setup_astrodino(
    astrodino_output_dir: str,
    astrodino_config_file: str,
    astrodino_pretrained_weights: str,
    print_dino: bool = False,
) -> torch.nn.Module:
    """Set up AstroDINO model"""

    # Set up config to pass to AstroDINO
    class config:
        output_dir = astrodino_output_dir
        config_file = astrodino_config_file
        pretrained_weights = astrodino_pretrained_weights
        opts = []

    if not print_dino:
        sys.stdout = open(os.devnull, "w")  # Redirect stdout to null
    astrodino, _ = setup_and_build_model(config())
    if not print_dino:
        sys.stderr = sys.__stderr__  # Reset stderr
    return astrodino


def get_embeddings(
    models: Dict[str, torch.nn.Module], images: torch.Tensor, batch_size: int = 64
) -> dict:
    """Get embeddings for images using models"""
    model_embeddings = {}
    batch_images = []

    for image in tqdm(images):
        # Load images, already preprocessed
        batch_images.append(torch.tensor(image, dtype=torch.float32)[None, :, :, :])

        # Get embeddings for batch
        if len(batch_images) == batch_size:
            with torch.no_grad():
                for key in models.keys():
                    model_embeddings[key] = (
                        models[key](torch.cat(batch_images).cuda()).cpu().numpy()
                    )
            batch_images = []

    # Get embeddings for last batch
    if len(batch_images) > 0:
        with torch.no_grad():
            for key in models.keys():
                model_embeddings[key] = (
                    models[key](torch.cat(batch_images).cuda()).cpu().numpy()
                )

    for key in model_embeddings.keys():
        model_embeddings[key] = np.concatenate(model_embeddings[key])

    return embeddings


def main(
    galaxy_zoo_file,
    astrodino_output_dir="/mnt/ceph/users/polymathic/astroclip/outputs/astroclip_image/u6lwxdfu/",
    astrodino_config_file="/astroclip/astrodino/config.yaml",
    astrodino_pretrained_weights="/mnt/ceph/users/polymathic/astroclip/pretrained/astrodino_newest.ckpt",
):
    # Set up models
    models = {
        "astrodino": setup_astrodino(
            astrodino_output_dir, astrodino_config_file, astrodino_pretrained_weights
        )
    }

    # Load data
    galaxy_zoo = Table.read(galaxy_zoo_file)
    images = galaxy_zoo["image"]

    # Get embeddings
    embeddings = get_embeddings(models, images)

    # Remove images and replace with embeddings
    galaxy_zoo.remove_column("image")
    for key in models.keys():
        galaxy_zoo[f"{key}_embeddings"] = embeddings[key]

    # Save embeddings
    galaxy_zoo.write(galaxy_zoo_file.replace(".h5", "_embeddings.h5"), overwrite=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--galaxy_zoo_file",
        type=str,
        default="/mnt/ceph/users/polymathic/astroclip/datasets/galaxy_zoo/gz5_decals_crossmatched.h5",
    )
    parser.add_argument(
        "--astrodino_output_dir",
        type=str,
        default="/mnt/ceph/users/polymathic/astroclip/outputs/astroclip_image/u6lwxdfu/",
    )
    parser.add_argument(
        "--astrodino_config_file", type=str, default="./astroclip/astrodino/config.yaml"
    )
    parser.add_argument(
        "--astrodino_pretrained_weights",
        type=str,
        default="/mnt/ceph/users/polymathic/astroclip/pretrained/astrodino_newest.ckpt",
    )
    args = parser.parse_args()

    main(
        args.galaxy_zoo_file,
        args.astrodino_output_dir,
        args.astrodino_config_file,
        args.astrodino_pretrained_weights,
    )
