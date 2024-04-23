import os
import h5py
import numpy as np
from tqdm import tqdm
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from typing import List

# Image Files locations
files_north = [os.path.join('/mnt/ceph/users/polymathic/external_data/astro/DECALS_Stein_et_al/north/', 'images_npix152_0%02d000000_0%02d000000.h5'%(i,i+1)) for i in range(10)]
files_south = [os.path.join('/mnt/ceph/users/polymathic/external_data/astro/DECALS_Stein_et_al/south/', 'images_npix152_0%02d000000_0%02d000000.h5'%(i,i+1)) for i in range(62)]

# Classifications location
gz5_decals_path = '/mnt/home/lparker/ceph/gz_decals_volunteers_5.csv'
gz2_sdss_path = '/mnt/home/lparker/ceph/gz2_hart16.csv'

def _generate_catalog(files: List[str]) -> Table:
    """Generate a catalog from a list of files."""
    ra_list, dec_list = [], []
    index_list, file_list = [], []
    print('Generating catalogs', flush=True)
    for i, file in enumerate(tqdm(files)):
        with h5py.File(file, 'r') as f:
            ra = f['ra'][:]
            dec = f['dec'][:]

            # Append data to lists
            ra_list.extend(ra)
            dec_list.extend(dec)
            file_list.extend([file]*len(ra))
            index_list.extend(range(0, len(ra)))

    # Create astropy table
    return Table([ra_list, dec_list, index_list, file_list], names=('ra', 'dec', 'index', 'file'))

def _cross_match_tables(table1: Table, table2: Table, max_sep: float = 0.5) -> tuple[Table, Table]:
    """Cross-match two tables."""

    # Create SkyCoord objects
    coords1 = SkyCoord(ra=table1['ra']*u.degree, dec=table1['dec']*u.degree)
    coords2 = SkyCoord(ra=table2['ra']*u.degree, dec=table2['dec']*u.degree)

    print('Matching coordinates', flush=True)

    # Match coordinates
    idx, d2d, _ = coords1.match_to_catalog_sky(coords2)

    # Define separation constraint and apply it
    max_sep = max_sep * u.arcsec
    sep_constraint = d2d < max_sep

    print(f'Total number of matches: {np.sum(sep_constraint)} \n', flush=True)
    return table1[sep_constraint], table2[idx[sep_constraint]]

def _get_images(files: list[str], classifications: Table) -> Table:
    """Get images from files."""
    print('Adding images to catalog', flush=True)

    classifications['image'] = np.zeros((len(classifications), 3, 152, 152))
    for file in tqdm(files):
        with h5py.File(file, 'r') as f:
            for k, entry in enumerate(classifications):
                if entry['file'] != file: continue
                index = entry['index']
                classifications[k]['image'] = f['images'][index]     

    return classifications

def _save_classifications(classifications: Table, path: str):
    """Save classifications to a path."""
    classifications.write(path, overwrite=True)

def get_paired_classifications(
        sky: str = 'south', 
        gz_survey: str = 'gz5',
        save_file: bool = True
    ) -> Table:
    """
    Pairs Galaxy Zoo classifications with DECaLS images in an Astropy table.

    Args:
        sky (str): Sky type, either 'south' or 'north'.
        gz_survey (str): GZ survey type, either 'gz2' or 'gz5'.
        save_file (bool): Whether to save the file.

    Returns:
        Table: Table of paired classifications.
    """

    if sky == 'south': files = files_south
    elif sky == 'north': files = files_north
    else: raise ValueError('Not supported sky type, choose south or north')
    
    if gz_survey == 'gz2': classifications_path = gz2_sdss_path
    elif gz_survey == 'gz5': classifications_path = gz5_decals_path
    else: raise ValueError('Not supported gz_survey type, choose gz2 or gz5')

    print(f'Sky type is {sky}, survey type is {gz_survey} \n', flush=True)
    
    # Load classifications
    morphologies = Table.read(classifications_path, format='ascii')
    
    # Generate catalog of ra, dec, index, file from files
    positions = _generate_catalog(files)
    
    # Cross-match positions with morphology classifications
    classifications, positions_matched = _cross_match_tables(morphologies, positions)
    
    # Update classifications with index and file
    classifications['index'] = np.array(positions_matched['index'])
    classifications['file'] = np.array(positions_matched['file'])
    
    # Get images and add them to classifications
    classifications = _get_images(files, classifications)

    # Save classifications
    if save_file: _save_classifications(classifications, classifications_path.replace('.csv', '_paired.ecsv'))

    return classifications