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

from aiohttp import ClientTimeout 
import datasets
import h5py
import numpy as np

_CITATION = """
"""

_DESCRIPTION = """\
This dataset is designed for cross-modal learning between images and spectra of galaxies
contained in the DESI Early Data Release and the Legacy Survey DR9. It contains roughly 150k
examples of images and spectra of galaxies, with their redshifts and targetids.
"""

_HOMEPAGE = ""

_LICENSE = ""

_URLS = {
    "joint": "https://users.flatironinstitute.org/~flanusse/astroclip_desi.1.1.5.h5",
}


class AstroClipDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.5")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="joint",
            version=VERSION,
            description="This part of the dataset covers examples from both specral and image domains",
        ),
    ]

    DEFAULT_CONFIG_NAME = "joint"

    def _info(self):
        if self.config.name == "joint":
            features = datasets.Features(
                {
                    "image": datasets.Array3D(shape=(152, 152, 3), dtype="float32"),
                    "spectrum": datasets.Array2D(shape=(7781, 1), dtype="float32"),
                    "redshift": datasets.Value("float32"),
                    "targetid": datasets.Value("int64"),
                }
            )
        else:
            raise NotImplementedError(
                "Only the joint configuration is implemented for now"
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.name]
        dl_manager.download_config.storage_options["timeout"] = ClientTimeout(total=5000, connect=1000)
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": data_dir, "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        with h5py.File(filepath) as d:
            for i in range(10):
                # Access the data
                images = d[str(i)]["images"]
                spectra = d[str(i)]["spectra"]
                redshifts = d[str(i)]["redshifts"]
                targetids = d[str(i)]["targetids"]

                dset_size = len(targetids)

                if split == "train":
                    dset_range = (0, int(0.8 * dset_size))
                else:
                    dset_range = (int(0.8 * dset_size), dset_size)

                for j in range(dset_range[0], dset_range[1]):
                    yield str(targetids[j]), {
                        "image": np.array(images[j]).astype("float32"),
                        "spectrum": np.reshape(spectra[j], [-1, 1]).astype("float32"),
                        "redshift": redshifts[j],
                        "targetid": targetids[j],
                    }
