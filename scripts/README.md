
## Dataset generation

The following scripts are used to generate the datasets used in the paper:

    - `cross_match_data.py`: Finds spectra for objects in the Legacy Survey
    data prepared by George Stein (https://github.com/georgestein/ssl-legacysurvey/tree/main)

    - `export_data.py`: Exports the combination of images and spectra into
    a single HDF5 file.

In principle you should not need to run these scripts, as the datasets are
already provided by the resulting HuggingFace datasets. However, these
scripts are provided for reproducibility purposes.
