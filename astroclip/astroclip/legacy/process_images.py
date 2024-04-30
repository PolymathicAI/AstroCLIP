import torch
from datasets import load_from_disk


def get_astropile_dataloaders(batch_size):
    # Load the dataset from Huggingface
    dataset = load_from_disk(
        "/mnt/ceph/users/polymathic/mmoma/datasets/astroclip_file/"
    )
    dataset.set_format(type="torch", columns=["image", "spectrum", "targetid"])

    # Create the dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset["test"],
        batch_size=batch_size,
        num_workers=16,
        shuffle=False,
        drop_last=True,
    )

    return train_dataloader, val_dataloader


train_dataloader, val_dataloader = get_astropile_dataloaders(10)

import numpy as np
from torchvision import transforms
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def sdss_rgb(imgs, bands, scales=None, m=0.02):
    rgbscales = {
        "u": (2, 1.5),  # 1.0,
        "g": (2, 2.5),
        "r": (1, 1.5),
        "i": (0, 1.0),
        "z": (0, 0.4),  # 0.3
    }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        img = torch.maximum(torch.tensor(0), img * scale + m)
        I = I + img
    I /= len(bands)
    Q = 20
    fI = torch.arcsinh(Q * I) / torch.sqrt(torch.tensor(Q))
    I += (I == 0.0) * 1e-6
    H, W = I.shape
    rgb = torch.zeros((H, W, 3)).to(torch.float32)
    for img, band in zip(imgs, bands):
        plane, scale = rgbscales[band]
        rgb[:, :, plane] = (img * scale + m) * fI / I
    rgb = torch.clip(rgb, 0, 1)
    return rgb


def dr2_rgb(rimgs, bands, **ignored):
    return sdss_rgb(
        rimgs, bands, scales=dict(g=(2, 6.0), r=(1, 3.4), z=(0, 2.2)), m=0.03
    )


class toRGB(transforms.ToTensor):
    def __init__(self, bands, scales=None, m=0.02):
        self.bands = bands
        self.scales = scales
        self.m = m

    def __call__(self, rimgs):
        if len(rimgs.shape) == 3:
            return dr2_rgb(rimgs.T, self.bands).T
        if len(rimgs.shape) == 4:
            img_outs = []
            for img in rimgs:
                img_outs.append(dr2_rgb(img.T, self.bands).T[None, :, :, :])
            return torch.concatenate(img_outs)


# Define transforms
to_rgb = toRGB(bands=["g", "r", "z"])
img_transforms = Compose(
    [Resize(152, InterpolationMode.BICUBIC), ToTensor(), CenterCrop(144)]
)

import PIL.Image as im
from tqdm import tqdm

# Iterate through the dataloader
images = []
spectra = []
targetids = []
for i, batch in enumerate(tqdm(train_dataloader)):
    image, sp, targetid = batch["image"], batch["spectrum"].squeeze(), batch["targetid"]

    image_batch = np.array(to_rgb(image) * 255).astype("uint8").transpose(0, 2, 3, 1)
    image_batch = torch.stack(
        [
            img_transforms(im.fromarray(image_batch[i]))
            for i in range(image_batch.shape[0])
        ]
    )

    images.append(image_batch)
    spectra.append(sp)
    targetids.append(targetid)

images = torch.cat(images)
spectra = torch.cat(spectra)
targetids = torch.cat(targetids)

torch.save(images, "/mnt/home/lparker/ceph/images_train.pt")
torch.save(spectra, "/mnt/home/lparker/ceph/spectra_train.pt")
torch.save(targetids, "/mnt/home/lparker/ceph/targetids_train.pt")

# Iterate through the dataloader
images = []
spectra = []
targetids = []
for i, batch in enumerate(tqdm(val_dataloader)):
    image, sp, targetid = batch["image"], batch["spectrum"].squeeze(), batch["targetid"]

    image_batch = np.array(to_rgb(image) * 255).astype("uint8").transpose(0, 2, 3, 1)
    image_batch = torch.stack(
        [
            img_transforms(im.fromarray(image_batch[i]))
            for i in range(image_batch.shape[0])
        ]
    )

    images.append(image_batch)
    spectra.append(sp)
    targetids.append(targetid)

images = torch.cat(images)
spectra = torch.cat(spectra)
targetids = torch.cat(targetids)

# TODO: needs refactoring
torch.save(images, "/mnt/home/lparker/ceph/images_val.pt")
torch.save(spectra, "/mnt/home/lparker/ceph/spectra_val.pt")
torch.save(targetids, "/mnt/home/lparker/ceph/targetids_val.pt")
