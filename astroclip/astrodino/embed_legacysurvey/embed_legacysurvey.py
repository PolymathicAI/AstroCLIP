import argparse
import os
from multiprocessing import Pool

import h5py
import numpy as np
import torch
from datasets import load_dataset
from torch import package
from torchvision.transforms import CenterCrop, Compose, ToTensor
from tqdm import tqdm

# Set up dataset
crop = CenterCrop(144)
RGB_SCALES = {
    "u": (2, 1.5),
    "g": (2, 6.0),
    "r": (1, 3.4),
    "i": (0, 1.0),
    "z": (0, 2.2),
}


def decals_to_rgb(image, bands=["g", "r", "z"], scales=None, m=0.03, Q=20.0):
    axes, scales = zip(*[RGB_SCALES[bands[i]] for i in range(len(bands))])
    scales = [scales[i] for i in axes]
    image = image.movedim(1, -1).flip(-1)
    scales = torch.tensor(scales, dtype=torch.float32).to(image.device)
    I = torch.sum(torch.clamp(image * scales + m, min=0), dim=-1) / len(bands)
    fI = torch.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.0) * 1e-6
    image = (image * scales + m) * (fI / I).unsqueeze(-1)
    image = torch.clamp(image, 0, 1)
    return image.movedim(-1, 1)


def import_package(path: str, device: str = "cpu") -> torch.nn.Module:
    """Import a torch package from a given path"""
    importer = package.PackageImporter(path)
    model = importer.load_pickle("network", "network.pkl", map_location=device)
    return model


def process_file(args) -> None:
    """Process a single file in the dataset"""
    file, save_dir, batch_size, gpu_id = args
    file_path = os.path.join(dset_root, file, "001-of-001.hdf5")

    # Set the GPU device for this process
    torch.cuda.set_device(gpu_id)

    # Load the model
    astrodino = import_package(
        "/mnt/ceph/users/polymathic/astroclip/pretrained/astrodino.pt"
    ).to(torch.device(f"cuda:{gpu_id}"))

    embeddings = []
    with h5py.File(file_path, "r") as f:
        img_batch = []
        for img in tqdm(f["image_array"]):
            # Convert to RGB
            img = crop(torch.tensor(img[[0, 1, 3]]))  # get g,r,z

            # Append to batch
            img_batch.append(img)

            if len(img_batch) == batch_size:
                with torch.no_grad():
                    images = torch.stack(img_batch).cuda()
                    images = decals_to_rgb(images)
                    emb = astrodino(images)
                    embeddings.append(emb.cpu().numpy())
                im_batch = []

        # Get ra, dec, obj_id
        ra = f["RA"][:]
        dec = f["DEC"][:]
        obj_id = f["object_id"][:]

    # Concatenate embeddings
    embeddings = np.concatenate(embeddings, axis=0)

    # Save embeddings
    save_dir = os.path.join(save_dir, file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, "001-of-001.hdf5")
    with h5py.File(save_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings)
        f.create_dataset("RA", data=ra)
        f.create_dataset("DEC", data=dec)
        f.create_dataset("object_id", data=obj_id)


def embed_legacysurvey(
    dset_root: str, save_dir: str, astrodino_dir: str, batch_size=512, num_gpus=4
):
    # List all files in the dataset
    files = os.listdir(dset_root)

    # Create arguments for each process
    args = [(f, save_dir, batch_size, i % num_gpus) for i, f in enumerate(files)]

    # Use multiprocessing to process files in parallel
    with Pool(processes=num_gpus) as pool:
        pool.map(process_file, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument(
        "--astrodino_dir",
        type=str,
        default="/mnt/ceph/users/polymathic/astroclip/pretrained",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_gpus", type=int, default=4)
    args = parser.parse_args()

    # Run the embedding process
    embed_legacysurvey(dset_root, save_dir, astrodino_dir, batch_size, num_gpus)
