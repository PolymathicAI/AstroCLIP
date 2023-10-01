# This script exports the data needed for the dataset into a single file.
#
import h5py
from astropy.table import Table, join
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR='/mnt/home/flanusse/ceph'

# Open matched catalog
joint_cat = pd.read_parquet(DATA_DIR+'/matched_catalog.pq').drop_duplicates(subset=["key"])

# Create randomized indices to shuffle the dataset
rng = np.random.default_rng(seed=42)
indices = rng.permutation(len(joint_cat))
joint_cat = joint_cat.iloc[indices]

with h5py.File(DATA_DIR+'/exported_data.h5', 'w') as f:
    for i in range(10):
        print("Processing file %d"%i)
        # Considering only the objects that are in the current file
        sub_cat = joint_cat[joint_cat['inds'] // 1000000 == i]
        images = []
        spectra = []
        redshifts = []
        targetids = []
        with h5py.File(DATA_DIR+'/images_npix152_0%02d000000_0%02d000000.h5'%(i,i+1)) as d:
            for j in tqdm(range(len(sub_cat))):
                images.append(np.array(d['images'][sub_cat['inds'].iloc[j] % 1000000]).T.astype('float32'))
                spectra.append(np.reshape(sub_cat['flux'].iloc[j], [-1, 1]).astype('float32'))
                redshifts.append(sub_cat['redshift'].iloc[j])
                targetids.append(sub_cat['targetid'].iloc[j])
        f.create_group(str(i))
        f[str(i)].create_dataset('images', data=images)
        f[str(i)].create_dataset('spectra', data=spectra)
        f[str(i)].create_dataset('redshifts', data=redshifts)
        f[str(i)].create_dataset('targetids', data=targetids)
