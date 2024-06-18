## Property Estimation
We demonstrate physical property estimation using the PROVABGS dataset from Hahn, et al. (2022). This includes best-fit parameters from galaxy spectroscopy of stellar mass, age, metallicity, and star formation rate generated with a state-of-the-art bayesian SED modeling framework.

### Cross-Matching
Cross-matching between PROVABGS and the DESI x DESI-LS dataset is performed by running
```python
python property_utils/cross_match.py
```
This creates a cross-matched table with containing the preprocessed DESI x DESI-LS survey images and spectra and their corresponding PROVABGS physical properties.

### Embedding
Embedding of the images and spectra is then performed with
```python
python embed_provabgs.py
```
This creates a table containing the embedded DESI x DESI-LS images and spectra and their corresponding PROVABGS physical properties. Note that this assumes that the cross-matching in the above step has already been performed. Additionally, it also assumes that all models have been downloaded and stored in the following directory:
```bash
{ASTROCLIP_ROOT}/pretrained
```

### Property Estimation
Once the embedded table has been generated:
- Redshift estimation is performed in `redshift.ipynb`
- Property estimation is performed in `property_estimation.ipynb`
- Posterior estimation is performed with `posterior_estimation.py`

### Baselines
The baseline models used in the paper are trained in the `baselines` directory. Refer to the README therein for more details.
