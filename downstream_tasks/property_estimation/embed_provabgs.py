import os
from argparse import ArgumentParser
from typing import Dict

import numpy as np
import torch
from astropy.table import Table
from dinov2.eval.setup import setup_and_build_model
from torchvision.transforms import CenterCrop, Compose
from tqdm import tqdm

from astroclip.astrodino.utils import setup_astrodino
from astroclip.env import format_with_env
from astroclip.models import AstroClipModel, Moco_v2, SpecFormer


def get_embeddings(
    image_models: Dict[str, torch.nn.Module],
    spectrum_models: Dict[str, torch.nn.Module],
    images: torch.Tensor,
    spectra: torch.Tensor,
    batch_size: int = 512,
) -> dict:
    """Get embeddings for images using models"""
    full_keys = set(image_models.keys()).union(spectrum_models.keys())
    model_embeddings = {key: [] for key in full_keys}
    im_batch, sp_batch = [], []

    assert len(images) == len(spectra)
    for image, spectrum in tqdm(zip(images, spectra)):
        # Load images, already preprocessed
        im_batch.append(torch.tensor(image, dtype=torch.float32)[None, :, :, :])
        sp_batch.append(torch.tensor(spectrum, dtype=torch.float32)[None, :, :])

        # Get embeddings for batch
        if len(im_batch) == batch_size:
            with torch.no_grad():
                spectra, images = torch.cat(sp_batch).cuda(), torch.cat(im_batch).cuda()

                for key in image_models.keys():
                    model_embeddings[key].append(image_models[key](images))

                for key in spectrum_models.keys():
                    model_embeddings[key].append(spectrum_models[key](spectra))

            im_batch, sp_batch = [], []

    # Get embeddings for last batch
    if len(im_batch) > 0:
        with torch.no_grad():
            spectra, images = torch.cat(sp_batch).cuda(), torch.cat(im_batch).cuda()

            # Get embeddings
            for key in image_models.keys():
                model_embeddings[key].append(image_models[key](images))

            for key in spectrum_models.keys():
                model_embeddings[key].append(spectrum_models[key](spectra))

    model_embeddings = {
        key: np.concatenate(model_embeddings[key]) for key in model_embeddings.keys()
    }
    return model_embeddings


def embed_provabgs(
    provabgs_file_train: str,
    provabgs_file_test: str,
    pretrained_dir: str,
    batch_size: int = 512,
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

    # Set up SpecFormer model
    checkpoint = torch.load(pretrained_weights["specformer"])
    specformer = SpecFormer(**checkpoint["hyper_parameters"])
    specformer.load_state_dict(checkpoint["state_dict"])
    specformer.cuda()

    # Set up AstroDINO model
    astrodino = setup_astrodino(astrodino_output_dir, pretrained_weights["astrodino"])

    # Set up model dict
    image_models = {
        "astrodino": lambda x: astrodino(x).cpu().numpy(),
        "stein": lambda x: stein(x).cpu().numpy(),
        "astroclip_image": lambda x: astroclip(x, input_type="image").cpu().numpy(),
    }

    spectrum_models = {
        "astroclip_spectrum": lambda x: astroclip(x, input_type="spectrum")
        .cpu()
        .numpy(),
        "specformer": lambda x: np.mean(
            specformer(x)["embedding"].cpu().numpy(), axis=1
        ),
    }
    print("Models are correctly set up!")

    # Load data
    files = [provabgs_file_test, provabgs_file_train]
    for f in files:
        provabgs = Table.read(f)
        images, spectra = provabgs["image"], provabgs["spectrum"]

        # Get embeddings
        embeddings = get_embeddings(
            image_models, spectrum_models, images, spectra, batch_size
        )

        # Remove images and replace with embeddings
        provabgs.remove_column("image")
        provabgs.remove_column("spectrum")
        for key in embeddings.keys():
            assert len(embeddings[key]) == len(provabgs), "Embeddings incorrect length"
            provabgs[f"{key}_embeddings"] = embeddings[key]

        # Save embeddings
        provabgs.write(f.replace(".hdf5", "_embeddings.hdf5"), overwrite=True)


if __name__ == "__main__":
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")
    parser = ArgumentParser()
    parser.add_argument(
        "--provabgs_file_train",
        type=str,
        default=f"{ASTROCLIP_ROOT}/datasets/provabgs/provabgs_paired_train.hdf5",
    )
    parser.add_argument(
        "--provabgs_file_test",
        type=str,
        default=f"{ASTROCLIP_ROOT}/datasets/provabgs/provabgs_paired_test.hdf5",
    )
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default=f"{ASTROCLIP_ROOT}/pretrained/",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    embed_provabgs(
        args.provabgs_file_train,
        args.provabgs_file_test,
        args.pretrained_dir,
        args.batch_size,
    )
