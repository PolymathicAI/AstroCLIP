import os
from argparse import ArgumentParser
from typing import Dict

import numpy as np
import torch
from astropy.table import Table
from dinov2.eval.setup import setup_and_build_model
from tqdm import tqdm

from astroclip.astrodino.utils import setup_astrodino
from astroclip.env import format_with_env
from astroclip.models import AstroClipModel, Moco_v2, SpecFormer


def get_embeddings(
    image_models: Dict[str, torch.nn.Module],
    images: torch.Tensor,
    batch_size: int = 512,
) -> dict:
    """Get embeddings for images using models"""
    model_embeddings = {key: [] for key in image_models.keys()}
    im_batch = []

    for image in tqdm(images):
        # Load images, already preprocessed
        im_batch.append(torch.tensor(image, dtype=torch.float32)[None, :, :, :])

        # Get embeddings for batch
        if len(im_batch) == batch_size:
            with torch.no_grad():
                images = torch.cat(im_batch).cuda()
                for key in image_models.keys():
                    model_embeddings[key].append(image_models[key](images))

            im_batch = []

    # Get embeddings for last batch
    if len(im_batch) > 0:
        with torch.no_grad():
            images = torch.cat(im_batch).cuda()
            for key in image_models.keys():
                model_embeddings[key].append(image_models[key](images))

    model_embeddings = {
        key: np.concatenate(model_embeddings[key]) for key in model_embeddings.keys()
    }
    return model_embeddings


def embed_galaxy_zoo(
    galaxy_zoo_file: str,
    pretrained_dir: str,
    batch_size: int = 128,
):
    # Get directories
    astrodino_output_dir = os.path.join(pretrained_dir, "astrodino_output_dir")

    pretrained_weights = {}
    for model in ["astroclip", "stein", "astrodino", "specformer"]:
        pretrained_weights[model] = os.path.join(pretrained_dir, f"{model}.ckpt")

    # Set up AstroCLIP
    astroclip = AstroClipModel.load_from_checkpoint(
        checkpoint_path=pretrained_weights["astroclip"],
    )

    # Set up Stein, et al. model
    stein = Moco_v2.load_from_checkpoint(
        checkpoint_path=pretrained_weights["stein"],
    ).encoder_q

    # Set up AstroDINO model
    astrodino = setup_astrodino(astrodino_output_dir, pretrained_weights["astrodino"])

    # Set up model dict
    image_models = {
        "astrodino": lambda x: astrodino(x).cpu().numpy(),
        "stein": lambda x: stein(x).cpu().numpy(),
        "astroclip": lambda x: astroclip(x, input_type="image").cpu().numpy(),
    }
    print("Models are correctly set up!")

    # Get embeddings
    galaxy_zoo = Table.read(galaxy_zoo_file)
    images = galaxy_zoo["image"]
    embeddings = get_embeddings(image_models, images, batch_size)

    # Remove images and replace with embeddings
    galaxy_zoo.remove_column("image")
    for key in embeddings.keys():
        assert len(embeddings[key]) == len(galaxy_zoo), "Embeddings incorrect length"
        galaxy_zoo[f"{key}_embeddings"] = embeddings[key]

    # Save embeddings
    galaxy_zoo.write(galaxy_zoo_file.replace(".h5", "_embeddings.h5"), overwrite=True)


if __name__ == "__main__":
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")
    parser = ArgumentParser()
    parser.add_argument(
        "--galaxy_zoo_file",
        type=str,
        default=f"{ASTROCLIP_ROOT}/datasets/galaxy_zoo/gz5_decals_crossmatched.h5",
    )
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default=f"{ASTROCLIP_ROOT}/pretrained",
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    embed_galaxy_zoo(
        args.galaxy_zoo_file,
        args.pretrained_dir,
        args.batch_size,
    )
