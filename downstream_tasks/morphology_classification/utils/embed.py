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


def setup_stein(
    stein_pretrained_weights: str,
) -> torch.nn.Module:
    """Set up Stein, et al. model"""
    return Moco_v2.load_from_checkpoint(
        checkpoint_path=stein_pretrained_weights,
    ).encoder_q


def get_embeddings(
    models: Dict[str, torch.nn.Module],
    images: torch.Tensor,
    batch_size: int = 512,
) -> dict:
    """Get embeddings for images using models"""
    model_embeddings = {key: [] for key in models.keys()}
    im_batch = []

    for image in tqdm(images):
        # Load images, already preprocessed
        im_batch.append(torch.tensor(image, dtype=torch.float32)[None, :, :, :])

        # Get embeddings for batch
        if len(im_batch) == batch_size:
            with torch.no_grad():
                # Embed images
                images = torch.cat(im_batch).cuda()
                model_embeddings["astrodino"].append(
                    models["astrodino"](images).cpu().numpy()
                )
                model_embeddings["stein"].append(
                    models["stein"](CenterCrop(96)(images)).cpu().numpy()
                )
                model_embeddings["astroclip_image"].append(
                    models["astroclip_image"](images, input_type="image").cpu().numpy()
                )
            im_batch = []

    # Get embeddings for last batch
    if len(im_batch) > 0:
        with torch.no_grad():
            # Embed images
            images = torch.cat(im_batch).cuda()
            model_embeddings["astrodino"].append(
                models["astrodino"](images).cpu().numpy()
            )
            model_embeddings["stein"].append(
                models["stein"](CenterCrop(96)(images)).cpu().numpy()
            )
            model_embeddings["astroclip_image"].append(
                models["astroclip_image"](images, input_type="image").cpu().numpy()
            )

        # Concatenate embeddings
        for key in model_embeddings.keys():
            model_embeddings[key] = np.concatenate(model_embeddings[key])

    return model_embeddings


def main(
    galaxy_zoo_file: str,
    pretrained_dir: str,
    batch_size: int = 128,
):
    # Get directories
    astrodino_pretrained_weights = os.path.join(pretrained_dir, "astrodino.ckpt")
    astrodino_output_dir = os.path.join(pretrained_dir, "astrodino_output_dir")
    stein_pretrained_weights = os.path.join(pretrained_dir, "stein.ckpt")
    astroclip_pretrained_weights = os.path.join(pretrained_dir, "astroclip.ckpt")
    specformer_pretrained_weights = os.path.join(pretrained_dir, "specformer.ckpt")

    # Set up models
    models = {
        "astrodino": setup_astrodino(
            astrodino_output_dir, astrodino_pretrained_weights
        ),
        "stein": setup_stein(stein_pretrained_weights),
        "astroclip_image": setup_astroclip(astroclip_pretrained_weights),
    }

    # Load data
    galaxy_zoo = Table.read(galaxy_zoo_file)
    images = galaxy_zoo["image"]

    # Get embeddings
    embeddings = get_embeddings(models, images, batch_size)
    assert len(embeddings[key]) == len(galaxy_zoo), "Embeddings incorrect length"

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
        "--pretrained_dir",
        type=str,
        default="/mnt/ceph/users/polymathic/astroclip/pretrained",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    main(
        args.galaxy_zoo_file,
        args.pretrained_dir,
        args.batch_size,
    )
