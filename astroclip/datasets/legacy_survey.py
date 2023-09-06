# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Joint dataset of DESI Legacy Survey and DESI Early Data Release."""

import json
import os
import pandas as pd
import glob
import h5py
from astropy.table import Table, join
import datasets
import numpy as np

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {A great new dataset},
author={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This new dataset is designed to for cross-modal contrastive learning between
images and spectra.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "joint": "https://huggingface.co/great-new-dataset-first_domain.zip",
}

class DesiSSL(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.2")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="joint", version=VERSION, description="This part of the dataset covers examples from both specral and image domains"),
        # datasets.BuilderConfig(name="second_domain", version=VERSION, description="This part of my dataset covers a second domain"),
    ]

    DEFAULT_CONFIG_NAME = "joint"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if self.config.name == "joint":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    # "sentence": datasets.Value("string"),
                    # "option1": datasets.Value("string"),
                    # "answer": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                    "image": datasets.Array3D(shape=(152, 152, 3), dtype='float32'),
                    "spectrum": datasets.Array2D(shape=(7781,1), dtype='float32'),
                    "redshift": datasets.Value("float32")
                }
            )
        else:  # This is an example to show how to have different features for "first_domain" and "second_domain"
            # features = datasets.Features(
            #     {
            #         "sentence": datasets.Value("string"),
            #         "option2": datasets.Value("string"),
            #         "second_domain_answer": datasets.Value("string")
            #         # These are the features of your dataset like images, labels ...
            #     }
            # )
            raise NotImplementedError("Only the joint configuration is implemented for now")
        
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        # data_dir = dl_manager.download_and_extract(urls)
        data_dir = '/mnt/home/flanusse/ceph'
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        # Open matched catalog
        joint_cat = pd.read_parquet(filepath+'/matched_catalog.pq').drop_duplicates(subset=["key"])

        # Depending on the split, we only consider a subset of the data
        if split == 'train':
            data_range = (0, 8)
        elif split == 'test':
            data_range = (8, 10)

        for i in range(*data_range):
            # Considering only the objects that are in the current file
            sub_cat = joint_cat[joint_cat['inds'] // 1000000 == i]
            
            with h5py.File(filepath+'/images_npix152_0%02d000000_0%02d000000.h5'%(i,i+1)) as d:
                for j in range(len(sub_cat)):
                    yield str(sub_cat['key'].iloc[j]), {
                        "image": np.array(d['images'][sub_cat['inds'].iloc[j] % 1000000]).T.astype('float32'),
                        "spectrum": np.reshape(sub_cat['flux'].iloc[j], [-1, 1]).astype('float32'),
                        "redshift": sub_cat['redshift'].iloc[j]
                    }
