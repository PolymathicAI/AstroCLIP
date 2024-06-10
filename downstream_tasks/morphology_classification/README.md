## Morphology Classification
We demonstrate morphology classification using the GalaxyZoo DECaLS dataset. In particular, we use the classifications from GZD-5 (Walmsley, et al. (2022)), which includes over 7.5 million volunteer response classifications for roughly 314,000 galaxies on a variety of questions, including morphological T-types, strong bars, arm curvature, etc. 

### Cross-Matching
Cross-matching between GalaxyZoo DECaLS and the full DESI-LS survey is performed by running
```python
python morphology_utils/cross_match.py 
```
This creates a cross-matched table with containing the preprocessed DESI-LS survey images and their corresponding GalaxyZoo DECaLS volunteer classifications. Note that this assumes that the Legacy Survey images have been downloaded and correctly formatted in
```bash
{ASTROCLIP_ROOT}/datasets/decals
```
If they are stored elsewhere, this can be specified using the `--root_dir` flag. 

### Embedding
The images are then embedded using 
```python
python embed_galaxy_zoo.py
```
This creates a table containing the embedded DESI-LS survey images and their corresponding GalaxyZoo DECaLS volunteer classifications. Note that this assumes that the cross-matching in the above step has already been performed. Additionally, it also assumes that all models have been downloaded and stored in the following directory:
```bash
{ASTROCLIP_ROOT}/pretrained
```

### Classification
Once the embedded table has been generated, classification is performed in the `morphology_classification.ipynb` notebook; see that file for more details.


