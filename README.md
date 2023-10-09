# AstroCLIP
Multimodal contrastive pretraining for astronomical data

<a href="https://arxiv.org/abs/2310.03024" style='vertical-align:middle; display:inline;'><img
							src="https://img.shields.io/badge/astro--ph.IM-arXiv%3A2310.03024-B31B1B.svg" class="plain" style="height:25px;" /></a>


The goal of this project is to demonstrate the ability of contrastive pre-training between two different kinds of astronomical data modalities (multi-band imaging, and optical spectra), to yield a meaningful embedding space which captures physical information about galaxies and is shared between both modalities. 

![image](assets/im_embedding.png)

## Results

We encourage you to take a look at our [NeurIPS 2023 AI4Science submission](https://arxiv.org/abs/2310.03024) (still under review) for a longer form description of our results, but here are the main takeaways:
 - Both image and spectra encoders are able to extract meaningful physical information from the input data.
 - The embeddings of both images and spectra are well aligned, allowing us to retrieve spectra that correspond to a given image, and vice-versa.

The notebook used to generate the plots of the paper can be found [here](notebooks/PaperPlots.ipynb).

Below is a visulatization of the learned embeddings, by taking the 2 first PCA components of spectra and image embeddings. As one can see, images and spectra discover similar main factors of variations.
![emb_pca](https://github.com/PolymathicAI/AstroCLIP/assets/861591/01475caa-8628-439b-8553-951074e287e2)



## Products: Datasets and Trained Models

### Dataset

As part of this project, we compile and make available a combined dataset of DESI Legacy Survey g,r,z images, and DESI Early Data Release spectra. These images are a subset of the [ssl-legacysurvey](https://github.com/georgestein/ssl-legacysurvey) sample compiled by @georgestein from the Legacy Survey DR9. Scripts used to match these datasets are available [here](scripts/cross_match_data.py).

For convenience, we provide a Hugging Face Datasets loading script which will automatically download the data needed and prepare the dataset on your computer.

```python
from datasets import load_dataset

# This downloads about 60 GB of data
dset = load_dataset('astroclip/datasets/legacy_survey.py')
```

For an example of getting started with this dataset, for example to simply predict redsfhit from the spectra, you can take a look at this notebook  [notebook](notebooks/dev/ConvolutionalPrototyping.ipynb).


### Training scripts and model weights 

**[Coming soon]**


## Requirements

This repo should only have basic pytorch and huggingface requirements. The following should install all that is needed (when run from this repository):

```bash
pip install .
```

