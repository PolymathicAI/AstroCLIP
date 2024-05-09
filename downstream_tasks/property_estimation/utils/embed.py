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

from astroclip.models import AstroClipModel
from models import Moco_v2


def setup_astroclip(
    astroclip_pretrained_weights: str,
) -> torch.nn.Module:
    """Set up AstroClip model"""
    return AstroClipModel.load_from_checkpoint(
        checkpoint_path=astroclip_pretrained_weights,
    )


def setup_astrodino(
    astrodino_output_dir: str,
    astrodino_config_file: str,
    astrodino_pretrained_weights: str,
) -> torch.nn.Module:
    """Set up AstroDINO model"""

    # Set up config to pass to AstroDINO
    class config:
        output_dir = astrodino_output_dir
        config_file = astrodino_config_file
        pretrained_weights = astrodino_pretrained_weights
        opts = []

    sys.stdout = open(os.devnull, "w")  # Redirect stdout to null
    astrodino, _ = setup_and_build_model(config())
    sys.stderr = sys.__stderr__  # Reset stderr
    return astrodino


def setup_stein(
    stein_pretrained_weights: str,
) -> torch.nn.Module:
    """Set up Stein model"""
    return Moco_v2.load_from_checkpoint(
        checkpoint_path=stein_pretrained_weights,
    ).encoder_q


def get_embeddings(
    models: Dict[str, torch.nn.Module], images: torch.Tensor, batch_size: int = 64
) -> dict:
    """Get embeddings for images using models"""
    model_embeddings = {key: [] for key in models.keys()}
    batch = []

    for image in tqdm(images):
        # Load images, already preprocessed
        batch.append(torch.tensor(image, dtype=torch.float32)[None, :, :, :])

        # Get embeddings for batch
        if len(batch) == batch_size:
            with torch.no_grad():
                images = torch.cat(batch).cuda()
                model_embeddings["astrodino"].append(
                    models["astrodino"](images).cpu().numpy()
                )
                model_embeddings["stein"].append(
                    models["stein"](CenterCrop(96)(images)).cpu().numpy()
                )
                model_embeddings["astroclip"].append(
                    models["astroclip"](images, input_type="image").cpu().numpy()
                )
            batch = []

    # Get embeddings for last batch
    if len(batch) > 0:
        with torch.no_grad():
            images = torch.cat(batch).cuda()
            model_embeddings["astrodino"].append(
                models["astrodino"](images).cpu().numpy()
            )
            model_embeddings["stein"].append(
                models["stein"](CenterCrop(96)(images)).cpu().numpy()
            )
            model_embeddings["astroclip"].append(
                models["astroclip"](images, input_type="image").cpu().numpy()
            )

        # Concatenate embeddings
        for key in model_embeddings.keys():
            model_embeddings[key] = np.concatenate(model_embeddings[key])

    return model_embeddings


def main(
    provabgs_file_train: str,
    provabgs_file_test: str,
    astrodino_output_dir: str,
    astrodino_config_file: str,
    astrodino_pretrained_weights: str,
    stein_pretrained_weights: str,
    astroclip_pretrained_weights: str,
    batch_size: int = 128,
):
    # Set up models
    models = {
        "astrodino": setup_astrodino(
            astrodino_output_dir, astrodino_config_file, astrodino_pretrained_weights
        ),
        "stein": setup_stein(stein_pretrained_weights),
        "astroclip": setup_astroclip(astroclip_pretrained_weights),
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
    parser.add_argument(
        "--stein_pretrained_weights",
        type=str,
        default="/mnt/ceph/users/polymathic/astroclip/pretrained/stein.ckpt",
    )
    parser.add_argument(
        "--astroclip_pretrained_weights",
        type=str,
        default="/mnt/ceph/users/polymathic/astroclip/pretrained/astroclip.ckpt",
    )
    args = parser.parse_args()

    main(
        args.provabgs_file_train,
        args.provabgs_file_test,
        args.astrodino_output_dir,
        args.astrodino_config_file,
        args.astrodino_pretrained_weights,
        args.stein_pretrained_weights,
        args.astroclip_pretrained_weights,
        args.batch_size,
    )
