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

from astroclip.models import AstroClipModel, SpecFormer
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
    astrodino_pretrained_weights: str,
    astrodino_config_file: str = "./astroclip/astrodino/config.yaml",
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


def setup_specformer(
    specformer_pretrained_weights: str,
) -> torch.nn.Module:
    checkpoint = torch.load(specformer_pretrained_weights)
    model = SpecFormer(**checkpoint["hyper_parameters"]).cuda()
    return model


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
    spectra: torch.Tensor,
    batch_size: int = 512,
) -> dict:
    """Get embeddings for images using models"""
    model_embeddings = {key: [] for key in models.keys()}
    im_batch, sp_batch = [], []

    for image, spectrum in tqdm(zip(images, spectra)):
        # Load images, already preprocessed
        im_batch.append(torch.tensor(image, dtype=torch.float32)[None, :, :, :])
        sp_batch.append(torch.tensor(spectrum, dtype=torch.float32)[None, :, :])

        # Get embeddings for batch
        if len(im_batch) == batch_size:
            with torch.no_grad():
                # Embed spectra
                spectra = torch.cat(sp_batch).cuda()
                model_embeddings["astroclip_spectrum"].append(
                    models["astroclip_spectrum"](spectra, input_type="spectrum")
                    .cpu()
                    .numpy()
                )
                model_embeddings["specformer"].append(
                    np.mean(
                        models["specformer"](spectra)["embedding"].cpu().numpy(), axis=1
                    )
                )

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
            im_batch, sp_batch = [], []

    # Get embeddings for last batch
    if len(im_batch) > 0:
        with torch.no_grad():
            # Embed spectra
            spectra = torch.cat(sp_batch).cuda()
            model_embeddings["astroclip_spectrum"].append(
                models["astroclip_spectrum"](spectra, input_type="spectrum")
                .cpu()
                .numpy()
            )
            model_embeddings["specformer"].append(
                np.mean(
                    models["specformer"](spectra)["embedding"].cpu().numpy(), axis=1
                )
            )

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
    provabgs_file_train: str,
    provabgs_file_test: str,
    pretrained_dir: str,
    batch_size: int = 512,
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
        "astroclip_spectrum": setup_astroclip(astroclip_pretrained_weights),
        "specformer": setup_specformer(specformer_pretrained_weights),
    }

    # Load data
    files = [provabgs_file_test, provabgs_file_train]
    for f in files:
        provabgs = Table.read(f)
        images, spectra = provabgs["image"], provabgs["spectrum"]

        # Get embeddings
        embeddings = get_embeddings(models, images, spectra, batch_size)

        # Remove images and replace with embeddings
        provabgs.remove_column("image")
        provabgs.remove_column("spectrum")
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
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default="/mnt/ceph/users/polymathic/astroclip/pretrained/",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    main(
        args.provabgs_file_train,
        args.provabgs_file_test,
        args.pretrained_dir,
        args.batch_size,
    )
