from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from astroclip.data.datamodule import AstroClipCollator, AstroClipDataloader
from astroclip.models.astroclip import AstroClipModel


def main(model_path: str, dataset_path: str, save_path: str, batch_size: int = 256):
    """Extract embeddings from the AstroClip model and save them to a file"""
    astroclip = AstroClipModel.load_from_checkpoint(model_path)

    # Get the dataloader
    loader = AstroClipDataloader(
        path=dataset_path,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=AstroClipCollator(),
        columns=["image", "spectrum"],
    )
    loader.setup("fit")

    # Set up validation loader
    val_loader = loader.val_dataloader()

    # Get the embeddings over the dataset
    im_embeddings, sp_embeddings, images, spectra = [], [], [], []
    with torch.no_grad():
        for batch_test in tqdm(val_loader):
            # Append the image and spectrum to the list
            images.append(batch_test["image"].permute(0, 3, 1, 2))
            spectra.append(batch_test["spectrum"])

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

    # Save the embeddings
    torch.save(
        {
            "image_features": np.concatenate(im_embeddings),
            "spectrum_features": np.concatenate(sp_embeddings),
            "images": np.concatenate(images),
            "spectra": np.concatenate(spectra),
        },
        save_path,
    )


if __name__ == "__main__":
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
        default="/mnt/ceph/users/polymathic/astroclip/outputs/astroclip-alignment/l1uwsr42/checkpoints/last.ckpt",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset",
        default="/mnt/ceph/users/polymathic/astroclip/datasets/astroclip_file/",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=256,
    )
    args = parser.parse_args()
    main(args.model_path, args.dataset_path, args.save_path, args.batch_size)
