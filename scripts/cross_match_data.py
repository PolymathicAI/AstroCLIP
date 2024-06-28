# This script will cross match the cutouts from the legacy survey
# cutouts with spectra from the DESI EDR
import glob

import h5py
import numpy as np
import pandas as pd
from astropy.table import Table, join, vstack
from dl import authClient as ac
from dl import queryClient as qc
from sparcl.client import SparclClient
from tqdm import tqdm

DATA_DIR = "/mnt/home/flanusse/ceph"

client = SparclClient()
inc = [
    "specid",
    "redshift",
    "flux",
    "ra",
    "dec",
    "wavelength",
    "spectype",
    "specprimary",
    "survey",
    "program",
    "targetid",
    "coadd_fiberstatus",
]


print("Retrieving all objects in the DESI data release...")
query = """
SELECT phot.targetid, phot.brickid, phot.brick_objid, phot.release, zpix.healpix
FROM desi_edr.photometry AS phot
INNER JOIN desi_edr.zpix ON phot.targetid = zpix.targetid
WHERE (zpix.coadd_fiberstatus = 0 AND zpix.sv_primary)
"""
cat = qc.query(sql=query, fmt="table")
print("done")
# Building search key based on brick ids
cat["key"] = [
    "%d_%d_%d" % (cat["release"][i], cat["brickid"][i], cat["brick_objid"][i])
    for i in range(len(cat))
]

merged_cat = None

# Looping over the downloaded image files
for file in tqdm(glob.glob(DATA_DIR + "/*.h5")):
    try:
        with h5py.File(file) as d:
            # search key
            d_key = np.array(
                [
                    "%d_%d_%d" % (d["release"][i], d["brickid"][i], d["objid"][i])
                    for i in range(len(d["brickid"]))
                ]
            )
            t = Table(data=[d["inds"][:], d_key], names=["inds", "key"])
    except:
        continue
    file_cat = join(cat, t, keys=["key"])
    file_cat["image_file"] = file
    file_cat.sort("healpix")

    # Retrieving spectra associated with this file
    target_ids = [int(i) for i in file_cat["targetid"]]
    records = None
    for i in tqdm(range(len(target_ids) // 500 + 1)):
        start = i * 500
        end = min((i + 1) * 500, len(target_ids) - 1)

        res = client.retrieve_by_specid(
            specid_list=target_ids[start:end], include=inc, dataset_list=["DESI-EDR"]
        )
        if records is None:
            records = Table.from_pandas(pd.DataFrame.from_records(res.records))
        else:
            r = Table.from_pandas(pd.DataFrame.from_records(res.records))
            records = vstack([records, r])

    # Merging catalogs
    file_cat = join(file_cat, records, keys=["targetid"])

    if merged_cat is None:
        merged_cat = file_cat
    else:
        merged_cat = vstack([merged_cat, file_cat])

    # Saving the results
    merged_cat.to_pandas().to_parquet("matched_catalog.pq")
