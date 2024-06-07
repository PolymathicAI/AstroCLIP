# AstroCLIP

The goal of this project is to demonstrate the ability of contrastive pre-training between two different kinds of astronomical data modalities (multi-band imaging, and optical spectra), to yield a meaningful embedding space which captures physical information about galaxies and is shared between both modalities.

![image](assets/im_embedding.png)


## Getting Started
TODO: Link tutorial notebook.

## Installation
The training and evaluation code requires PyTorch 2.0. Additionally, an up-to-date eventlet is required for wandb. Note that teh code has only been tested with the specified versions and also expects a Linux environment. To install the AstroCLIP package and its corresponding dependencies, please follow the code below.

The following packages are excluded from the project's dependencies to allow for a more flexible system configuration (i.e. allow the use of module subsystem).

```bash
pip install --upgrade pip
pip install --upgrade eventlet torch lightning[extra]
pip install -e .
```

It is possible to override default storage path by changing the flag in `astroclip/env.py`

## Training

AstroCLIP is trained using a two-step process.

First, we pre-train a single-modal galaxy image encoder and a single-modal galaxy spectrum encoder separately.

### Image encoder:
The AstroDINO model is based on the DINO_v2 model and can be run from the astrodino subdirectory.

Run with
```
image_trainer -c astroclip/astrodino/config.yaml
```

### Spectrum encoder:

Run with
```
spectrum_trainer fit -c config/specformer.yaml

```

## Training alignment model

AstroCLIP model can be run with:
```
spectrum_trainer fit -c config/astroclip.yaml
```

## Downstream Tasks

TODO
