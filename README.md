# AstroCLIP

![image](assets/im_embedding.png)

## Install
Note that an up-to-date eventlet is required for wandb.
The following packages are excluded from the project's dependencies to allow for a more flexible system configuration (i.e. allow the use of module subsystem).

```bash
pip install --upgrade pip
pip install --upgrade eventlet torch lightning[extra]
pip install -e .
```


## Training Single-Modal SSL Models

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
