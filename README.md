# AstroCLIP
<a href="https://arxiv.org/abs/2310.03024" style='vertical-align:middle; display:inline;'><img
							src="https://img.shields.io/badge/astro--ph.IM-arXiv%3A2310.03024-B31B1B.svg" class="plain" style="height:25px;" /></a>

Official PyTorch implementation and pre-trained models for the paper **AstroCLIP: A Cross-Modal Foundation Model for Galaxies**.

![image](assets/im_embedding.png)

AstroCLIP is a novel, cross-modal, self-supervised foundation model that creates a shared embedding space for multi-band imaging and optical spectra of galaxies. These embeddings encode meaningful physical information shared between both modalities, and can be used as the basis for competitive zero- and few-shot learning on a variety of downstream tasks, including similarity search, redshift estimation, galaxy property prediction, and morphology classification.

## Tutorial
Check out a collab-native tutorial available on AstroCLIP here:
https://colab.research.google.com/github/EiffL/Tutorials/blob/master/FoundationModels/AstroCLIPTutorial_solutions.ipynb

## Web App
Check out our interactive similarity search app, enabling both in-modal and cross-modal search for galaxies:
https://astroclip.streamlit.app/

## Installation
The training and evaluation code requires PyTorch 2.0. Additionally, an up-to-date eventlet is required for wandb. Note that the code has only been tested with the specified versions and also expects a Linux environment. To install the AstroCLIP package and its corresponding dependencies, please follow the code below. The install procedure is unfortunately a bit complex, but execute these lines in a clean python environment should work.

```bash
# Setting up proper torch version
pip install --upgrade pip
pip install lightning[extra]==2.3.3 boto3==1.28.17 
pip install --upgrade pycairo datasets pyarrow
pip install --extra-index-url https://pypi.nvidia.com cuml-cu11
pip install --extra-index-url https://download.pytorch.org/whl/cu117 torch==2.0.0+cu117
pip install torchvision==0.15.0 torchmetrics==0.10.3 dotenv numpy==1.26.4

# Installing DiNOv2
pip install omegaconf fvcore iopath
pip install --no-deps git+https://github.com/facebookresearch/dinov2.git@2302b6bf46953431b969155307b9bed152754069

# Installing AstroCLIP
pip install astropy datasets huggingface_hub jaxtyping wandb
pip install --no-deps git+https://github.com/PolymathicAI/AstroCLIP.git
```
**NOTE** The package provides the three shortcuts: `astroclip_trainer` and `spectrum_trainer`, which link to `astroclip/trainer.py`, and `image_trainer`, which links to `astroclip/astrodino/trainer.py`, as long as it is installed. The shortcuts are defined in the `project.scripts` section of the `pyproject.toml` file.

#### Handling roots
The package expects to load models and data by default from
```bash
{ASTROCLIP_ROOT}
```

You can configure `ASTROCLIP_ROOT` as well as the weights and biases group in which runs are saved by creating a `.env` file in the root of `astroclip` with the following content:

```bash
ASTROCLIP_ROOT="/mnt/ceph/users/polymathic/astroclip"
WANDB_ENTITY_NAME="flatiron-scipt"
```

If no environment is specified, the default path at Flatiron will be assumed.

## Pretrained Models

We provide the pretrained AstroCLIP model on the Huggingface model hub for easy access. Additionally, we provide the pretrained single-modal models for galaxy images and spectra as well. Model details, checkpoints, configs and logs are below.

<table>
  <tr>
    <th>Model Name</th>
    <th>Pretraining</th>
    <th># Params.</th>
    <th colspan="3">Download</th>
  </tr>
  <tr>
    <td>AstroCLIP</td>
    <td>CLIP</td>
    <td>370M</td>
    <td><a href="https://huggingface.co/polymathic-ai/astroclip">ckpt</a></td>
    <td><a href="https://github.com/PolymathicAI/AstroCLIP/blob/main/configs/astroclip.yaml">config</a></td>
    <td><a href="https://example.com/link3">logs</a></td>
  </tr>
  <tr>
    <td>Image Encoder</td>
    <td>DINOv2</td>
    <td>302M</td>
    <td><a href="https://huggingface.co/polymathic-ai/astrodino">ckpt</a></td>
    <td><a href="https://github.com/PolymathicAI/AstroCLIP/blob/main/astroclip/astrodino/config.yaml">config</a></td>
    <td><a href="https://example.com/link3">logs</a></td>
  </tr>
  <tr>
    <td>Spectrum Encoder</td>
    <td>Masked Modeling</td>
    <td>43M</td>
    <td><a href="https://huggingface.co/polymathic-ai/specformer">ckpt</a></td>
    <td><a href="https://github.com/PolymathicAI/AstroCLIP/blob/main/configs/specformer.yaml">config</a></td>
    <td><a href="https://example.com/link3">logs</a></td>
  </tr>
</table>

#### Loading the Pretrained Models
The pretrained AstroCLIP model can be loaded using the following:
```python
from astroclip.models import AstroClipModel
model = AstroClipModel.load_from_checkpoint(
    checkpoint_path = "path_to_model.ckpt",
)
```

#### High-Level Performance Overview

Below, we include a high-level performance overview of our models on a variety of downstream tasks. This is non-exhaustive, and we refer the reader to the paper for the full details.

<table>
  <tr>
    <th>Source</th>
    <th>Model</th>
    <th>Type</th>
    <th>Redshift</th>
    <th>Properties</th>
    <th>Morphology</th>
  </tr>
  <tr>
    <td>Image</td>
    <td>AstroCLIP*</td>
    <td>Zero-Shot</td>
    <td>0.79</td>
    <td>0.47</td>
    <td>0.76</td>
  </tr>
  <tr>
    <td></td>
    <td>Image Encoder*</td>
    <td>Zero-Shot</td>
    <td>0.63</td>
    <td>0.37</td>
    <td>0.78</td>
  </tr>
  <tr>
    <td></td>
    <td>Stein, et al.</td>
    <td>Zero-Shot</td>
    <td>0.36</td>
    <td>0.26</td>
    <td>0.76</td>
  </tr>
  <tr>
    <td></td>
    <td>ResNet18</td>
    <td>Supervised</td>
    <td>0.77</td>
    <td>0.43</td>
    <td>-</td>
  </tr>
  <tr>
    <td></td>
    <td>ZooBot<sup>1</sup></td>
    <td>Supervised</td>
    <td>-</td>
    <td>-</td>
    <td>0.88</td>
  </tr>
  <tr>
    <td>Spectrum</td>
    <td>AstroCLIP*</td>
    <td>Zero-Shot</td>
    <td>0.99</td>
    <td>0.63</td>
    <td>-</td>
  </tr>
  <tr>
    <td></td>
    <td>Spectrum Encoder*</td>
    <td>Zero-Shot</td>
    <td>0.99</td>
    <td>0.64</td>
    <td>-</td>
  </tr>
  <tr>
    <td></td>
    <td>Conv+Att<sup>2</sup></td>
    <td>Supervised</td>
    <td>0.99</td>
    <td>0.60</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Photometry</td>
    <td>MLP</td>
    <td>Supervised</td>
    <td>0.68</td>
    <td>0.42</td>
    <td>-</td>
  </tr>
  <tr>
</table>

We report R-squared metrics on redshift and galaxy property estimation (averaged across all properties) and accuracy on galaxy morphology classification (averaged across all labels). Our models are marked with an asterisk (*). [1] We use the results reported from [Walmsley, et al. (2021)](https://github.com/mwalmsley/zoobot/tree/main). [2] We use the encoder from [Melchior, et al. (2022)](https://github.com/pmelchior/spender).

## Data Access

The AstroCLIP model is trained on the cross-matched sample containing optical spectra from the [Dark Energy Spectroscopic Instrument (DESI)](https://www.desi.lbl.gov/) Early Data Release (EDR) and multi-band images (g,r,z) from the [DESI Legacy Survey](https://www.legacysurvey.org/) prepared by [Stein, et al. (2022)](https://github.com/georgestein/ssl-legacysurvey/tree/main). We provide the dataset as a HuggingFace dataset, which can be accessed directly using

```python
from datasets import load_dataset

# This downloads about 60 GB of data
dset = load_dataset('astroclip/data/dataset.py')
```

For reproducibility, we include the scripts and a brief description of how to generate the cross-matched dataset in `astroclip/data/crossmatch`.

### Image Pretraining Dataset

![image](assets/decals.png)

While the AstroCLIP and Spectrum Encoder models are trained on the image-spectrum dataset, we pretrain the galaxy image model separately on full Stein, et al. (2022) image dataset, which consists of 76M galaxy images. This dataset can be accessed using this globus endpoint:

https://app.globus.org/file-manager?origin_id=9fb0fc0e-e760-11ec-9bd2-2d2219dcc1fa&origin_path=%2F

The directory is organized into south and north surveys, where each survey is split into chunks of 1,000,000 galaxies (sorted by decreasing z-band flux) and saved in hdf5 format. For more details, see [here](https://github.com/georgestein/ssl-legacysurvey/tree/main).


## Pretraining

AstroCLIP is trained using a two-step process:

1. We pre-train a single-modal galaxy image encoder and a single-modal galaxy spectrum encoder separately.
2. We CLIP-align these two encoders on a paired image-spectrum dataset.

### Single-Modal Pretraining

#### Image Pretraining - DINOv2 ViT:
AstroCLIP uses a Vision Transformer (ViT) to encode galaxy images. Pretraining is performed using the [DINOv2](https://github.com/facebookresearch/dinov2/) package, which combines self-distillation, masked-modeling, and contrastive objectives. Overall, we use largely the same training regime, however we modify some of the contrastive augmentations to suit an astrophysics context. Model training can be launched with the following command:
```
image_trainer -c astroclip/astrodino/config.yaml
```
We train the model using 20 A100 GPUs (on 5 nodes) for 250k steps which takes roughly 46 hours.

#### Spectrum Pretraining - Masked Modelling Transformer:
AstroCLIP uses a 1D Transformer to encode galaxy spectra. Pretraining is performed using a masked-modeling objective, whereby the 1D spectrum is split into contiguous, overlapping patches. Model training can be launched with the following command:
```
spectrum_trainer fit -c config/specformer.yaml
```
We train the model using 4 A100 GPUs (on 1 node) for 30k steps which takes roughly 12 hours.

### CLIP Alignment:

Once pretrained, we align the image and spectrum encoder using cross-attention projection heads to maximize the similarity between cross-modal embeddings that correspond to the same galaxy while simultaneously minimizing the similarity between cross-modal embeddings that correspond to different galaxies. Model training can be launched with the following command:
```
spectrum_trainer fit -c config/astroclip.yaml
```
We train the model using 4 A100 GPUs (on 1 node) for 25k steps or until the validation loss does not increase for a fixed number of steps. This takes roughly 12 hours.

## Downstream Tasks

We demonstrate that the AstroCLIP can be used to easily perform a variety of downstream tasks. In particular, we demonstrate their ability to do:

1. In-modal and cross-modal similarity search
2. Photometric redshift prediction
3. Physical property estimation from images
4. Physical property estimation from spectra
5. Morphology classification from images

The details of these downstream tasks and the results in our paper can be found in `astroclip/downstream_tasks`.

## Acknowledgements
This reposity uses datasets and contrastive augmentations from [Stein, et al. (2022)](https://github.com/georgestein/ssl-legacysurvey/tree/main). The image pretraining is built on top of the [DINOv2](https://github.com/facebookresearch/dinov2/) framework; we also thank Piotr Bojanowski for valuable conversations around image pretraining.

## License
AstroCLIP code and model weights are released under the MIT license. See [LICENSE](https://github.com/PolymathicAI/AstroCLIP/blob/main/LICENSE) for additional details.

## Citation
@article{Parker_2024,
   title={AstroCLIP: a cross-modal foundation model for galaxies},
   volume={531},
   ISSN={1365-2966},
   url={http://dx.doi.org/10.1093/mnras/stae1450},
   DOI={10.1093/mnras/stae1450},
   number={4},
   journal={Monthly Notices of the Royal Astronomical Society},
   publisher={Oxford University Press (OUP)},
   author={Parker, Liam and Lanusse, Francois and Golkar, Siavash and Sarra, Leopoldo and Cranmer, Miles and Bietti, Alberto and Eickenberg, Michael and Krawezik, Geraud and McCabe, Michael and Morel, Rudy and Ohana, Ruben and Pettee, Mariel and Régaldo-Saint Blancard, Bruno and Cho, Kyunghyun and Ho, Shirley},
   year={2024},
   month=jun, pages={4990–5011} }

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lhparker1"><img src="https://avatars.githubusercontent.com/u/86175266?v=4?s=100" width="100px;" alt="Liam Parker"/><br /><sub><b>Liam Parker</b></sub></a><br /><a href="https://github.com/PolymathicAI/AstroCLIP/commits?author=lhparker1" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://flanusse.net/"><img src="https://avatars.githubusercontent.com/u/861591?v=4?s=100" width="100px;" alt="Francois Lanusse"/><br /><sub><b>Francois Lanusse</b></sub></a><br /><a href="https://github.com/PolymathicAI/AstroCLIP/commits?author=EiffL" title="Code">💻</a> <a href="#data-EiffL" title="Data">🔣</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/golkar"><img src="https://avatars.githubusercontent.com/u/35383824?v=4?s=100" width="100px;" alt="Siavash Golkar"/><br /><sub><b>Siavash Golkar</b></sub></a><br /><a href="https://github.com/PolymathicAI/AstroCLIP/commits?author=golkar" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://users.flatironinstitute.org/~lsarra/"><img src="https://avatars.githubusercontent.com/u/66411731?v=4?s=100" width="100px;" alt="Leopoldo Sarra"/><br /><sub><b>Leopoldo</b></sub></a><br /><a href="https://github.com/PolymathicAI/AstroCLIP/commits?author=lsarra" title="Code">💻</a> <a href="#tool-lsarra" title="Tools">🔧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/shirleysurelyho"><img src="https://avatars.githubusercontent.com/u/3279839?v=4?s=100" width="100px;" alt="Shirley Ho"/><br /><sub><b>Shirley Ho</b></sub></a><br /><a href="#ideas-shirleysurelyho" title="Ideas, Planning, & Feedback">🤔</a> <a href="#fundingFinding-shirleysurelyho" title="Funding Finding">🔍</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MilesCranmer"><img src="https://avatars.githubusercontent.com/u/7593028?v=4?s=100" width="100px;" alt="Miles Cranmer"/><br /><sub><b>Miles Cranmer</b></sub></a><br /><a href="#ideas-MilesCranmer" title="Ideas, Planning, & Feedback">🤔</a> <a href="#design-MilesCranmer" title="Design">🎨</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
