# AstroCLIP

Official PyTorch implementation and pre-trained models for paper **AstroCLIP: A Cross-Modal Foundation Model for Galaxies**. 

![image](assets/im_embedding.png)

AstroCLIP is a novel, cross-modal, self-supervised foundation model that creates a shared embedding space for multi-band imaging and optical spectra of galaxies. These embeddings encode meaningful physical information shared between both modalities, and can be used as the basis for competitive zero- and few-shot learning on a variety of downstream tasks, including similarity search, redshift estimation, galaxy property prediction, and morphology classification. 

## Installation
The training and evaluation code requires PyTorch 2.0. Additionally, an up-to-date eventlet is required for wandb. Note that the code has only been tested with the specified versions and also expects a Linux environment. To install the AstroCLIP package and its corresponding dependencies, please follow the code below.

```bash
pip install --upgrade pip
pip install --upgrade eventlet torch lightning[extra]
pip install -e .
```

## Training

AstroCLIP is trained using a two-step process. First, we pre-train a single-modal galaxy image encoder and a single-modal galaxy spectrum encoder separately. Then, we CLIP align these two encoders on a paired image-spectrum dataset.

### DINOv2 ViT Image Pretraining:
AstroCLIP uses a Vision Transformer (ViT) to encode galaxy images. Pretraining is performed using the [DINOv2](https://github.com/facebookresearch/dinov2/tree/2302b6bf46953431b969155307b9bed152754069) package, which combines self-distillation, masked-modeling, and contrastive objectives. Overall, we use largely the same training regime, however we modify some of the contrastive augmentations to suite an astrophysics context. 

Model training can be launched with the following command:
```
image_trainer -c astroclip/astrodino/config.yaml
```
Ultimately, we run training using 20 A100 GPUs (on 5 nodes) for 250k steps using the config provided [here](https://github.com/PolymathicAI/AstroCLIP_v2/blob/master/astroclip/astrodino/config.yaml), which takes roughly 46 hours. 

### Spectrum encoder:

Run with
```
spectrum_trainer fit -c config/specformer.yaml

```

### CLIP alignment:

AstroCLIP model can be run with:
```
spectrum_trainer fit -c config/astroclip.yaml
```

## Downstream Tasks

TODO
