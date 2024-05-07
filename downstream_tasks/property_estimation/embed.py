import os
import sys
from argparse import ArgumentParser
from typing import Dict

import numpy as np
import torch
from astropy.table import Table
from dinov2.eval.setup import setup_and_build_model
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm


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
    model_embeddings = {key: [] for key in models.keys()}
    batch_images = []

    for image in tqdm(images):
        # Load images, already preprocessed
        batch_images.append(torch.tensor(image, dtype=torch.float32)[None, :, :, :])

        # Get embeddings for batch
        if len(batch_images) == batch_size:
            with torch.no_grad():
                for key in models.keys():
                    model_embeddings[key].append(
                        models[key](torch.cat(batch_images).cuda()).cpu().numpy()
                    )

            batch_images = []

    # Get embeddings for last batch
    if len(batch_images) > 0:
        with torch.no_grad():
            for key in models.keys():
                model_embeddings[key].append(
                    models[key](torch.cat(batch_images).cuda()).cpu().numpy()
                )
                model_embeddings[key] = np.concatenate(model_embeddings[key])

    return model_embeddings


def main(
    provabgs_file_train: str,
    provabgs_file_test: str,
    batch_size: int = 128,
    astrodino_output_dir: str = "/mnt/ceph/users/polymathic/astroclip/outputs/astroclip_image/u6lwxdfu/",
    astrodino_config_file: str = "/astroclip/astrodino/config.yaml",
    astrodino_pretrained_weights: str = "/mnt/ceph/users/polymathic/astroclip/pretrained/astrodino_newest.ckpt",
):
    # Set up models
    models = {
        "astrodino": setup_astrodino(
            astrodino_output_dir, astrodino_config_file, astrodino_pretrained_weights
        )
    }

    # Load data
    files = [provabgs_file_train, provabgs_file_test]
    for f in files:
        provabgs = Table.read(f)
        images = provabgs["image"]

        # Get embeddings
        embeddings = get_embeddings(models, images, batch_size)

        # Remove images and replace with embeddings
        provabgs.remove_column("image")
        for key in models.keys():
            assert len(embeddings[key]) == len(provabgs), "Embeddings incorrect length"
            provabgs[f"{key}_embeddings"] = embeddings[key]

        # Save embeddings
        provabgs.write(f.replace(".hdf5", "_embeddings.hdf5"), overwrite=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--provabgs_file_train",
        type=str,
        default="/mnt/ceph/users/polymathic/astroclip/datasets/provabgs/provabgs_paired_train.hdf5",
    )
    parser.add_argument(
        "--provabgs_file_test",
        type=str,
        default="/mnt/ceph/users/polymathic/astroclip/datasets/provabgs/provabgs_paired_test.hdf5",
    )
    parser.add_argument("--batch_size", type=int, default=128)
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
        args.provabgs_file_train,
        args.provabgs_file_test,
        args.batch_size,
        args.astrodino_output_dir,
        args.astrodino_config_file,
        args.astrodino_pretrained_weights,
    )
