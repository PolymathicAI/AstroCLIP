from argparse import ArgumentParser

import h5py
import numpy as np
import torch
from tqdm import tqdm

from astroclip.data.datamodule import AstroClipCollator, AstroClipDataloader
from astroclip.env import format_with_env
from astroclip.models.astroclip import AstroClipModel


def embed_astroclip(
    model_path: str,
    dataset_path: str,
    save_path: str,
    max_size: int = None,
    batch_size: int = 256,
    loader_type: str = "val",
):
    """Extract embeddings from the AstroClip model and save them to a file"""
    # Load the model
    astroclip = AstroClipModel.load_from_checkpoint(model_path)

    # Get the dataloader
    loader = AstroClipDataloader(
        path=dataset_path,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=AstroClipCollator(),
        columns=["image", "spectrum", "targetid"],
    )
    loader.setup("fit")

    # Set up loader
    if loader_type == "train":
        loader = loader.train_dataloader()
    elif loader_type == "val":
        loader = loader.val_dataloader()
    else:
        raise ValueError("loader must be either 'train' or 'val'")

    # Get the embeddings over the dataset
    im_embeddings, sp_embeddings, images, spectra, obj_ids = [], [], [], [], []
    with torch.no_grad():
        for idx, batch_test in tqdm(enumerate(loader), desc="Extracting embeddings"):
            # Break if max_size is reached
            if max_size is not None and idx * batch_size >= max_size:
                break

            # Append the image and spectrum to the list
            obj_ids.append(batch_test["targetid"])

            # Extract the embeddings
            im_embeddings.append(
                astroclip(batch_test["image"].cuda(), input_type="image")
                .detach()
                .cpu()
                .numpy()
            )
            sp_embeddings.append(
                astroclip(batch_test["spectrum"].cuda(), input_type="spectrum")
                .detach()
                .cpu()
                .numpy()
            )
            images.append(batch_test["image"])
            spectra.append(batch_test["spectrum"])

    # Save as an HDF5 file
    with h5py.File(save_path, "w") as f:
        f.create_dataset("image_embeddings", data=np.concatenate(im_embeddings))
        f.create_dataset("spectrum_embeddings", data=np.concatenate(sp_embeddings))
        f.create_dataset("object_id", data=np.concatenate(obj_ids))
        f.create_dataset("image", data=np.concatenate(images))
        f.create_dataset("spectrum", data=np.concatenate(spectra))
    print(f"Embeddings saved to {save_path}")


if __name__ == "__main__":
    ASTROCLIP_ROOT = format_with_env("{ASTROCLIP_ROOT}")
    parser = ArgumentParser()
    parser.add_argument(
        "save_path",
        type=str,
        help="Path to save the embeddings",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model",
        default=f"{ASTROCLIP_ROOT}/pretrained/astroclip.ckpt",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
        default=f"{ASTROCLIP_ROOT}/datasets/astroclip_file/",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=256,
    )
    parser.add_argument(
        "--max_size",
        type=int,
        help="Maximum number of samples to use",
        default=None,
    )
    parser.add_argument(
        "--loader_type",
        type=str,
        help="Which loader to use (train or val)",
        default="val",
    )
    args = parser.parse_args()
    embed_astroclip(
        args.model_path,
        args.dataset_path,
        args.save_path,
        args.max_size,
        args.batch_size,
        args.loader_type,
    )
