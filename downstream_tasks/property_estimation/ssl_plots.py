import os
import sys
import argparse

from astropy.table import Table, join
from datasets import load_dataset
import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from math import pi
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.helpers import zero_shot_train, few_shot_train, plot_radar, resnet_r2, photometry_r2, spender_r2
from utils.models import SimpleMLP

sns.set()
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})

# Overall Definitions:
properties = ['Z_HP', 'LOG_MSTAR', 'Z_MW', 't_ageMW', 'SFR']
property_titles = ['$Z_{HP}$', '$\log_{M_\star}$', '$\log Z_{MW}$', '$t_{age}$', '$\log sSFR$']
scaler = StandardScaler()

prop_dict = {
    'Z_HP': {
        'xlabel': 'True Redshift',
        'ylabel': 'Predicted Redshift',
        'lim_range': (0, 0.65),
        'text_position': (0.05, 0.55)
    },
    'LOG_MSTAR': {
        'xlabel': 'PROVABGS Log Stellar Mass $[M_\\odot]$',
        'ylabel': 'k-NN Predicted Log Stellar Mass $[M_\\odot]$',
        'lim_range': (9, 12.5),
        'text_position': (9.2, 12)
    },
    'Z_MW': {
        'xlabel': 'PROVABGS Metallicity $[Z_{mw}]$',
        'ylabel': 'k-NN Predicted Metallicity $[Z_{mw}]$',
        'lim_range': (-8, -4),
        'text_position': (np.log(0.005), np.log(0.03))
    },
    't_ageMW': {
        'xlabel': 'PROVABGS Galaxy Age $Gyr$',
        'ylabel': 'k-NN Predicted Galaxy Age $Gyr$',
        'lim_range': (0, 12),
        'text_position': (2, 12)
    },
    'SFR': {
        'xlabel': 'PROVABGS LogSpecific Star-Forming Rate $M_{\odot}/yr $',
        'ylabel': 'k-NN Star-Forming Region $M_{\odot}/yr$',
        'lim_range': (-15, -4),
        'text_position': (2, 12)
    }
}

# ----- Replace with new Dataset Loader ----- #
def get_provabgs(embedding_file, images=True):
    
    # Retrieve your CLIP embeddings
    CLIP_embeddings = h5py.File(embedding_file, 'r') 
    train_embeddings = CLIP_embeddings['train']
    test_embeddings  = CLIP_embeddings['test']
    
    if images:
        train_table = Table({'targetid': train_embeddings['targetid'], 
                             'image_features': train_embeddings['image_features']})

        test_table  = Table({'targetid': test_embeddings['targetid'], 
                             'image_features': test_embeddings['image_features']})
    
    else:
        train_table = Table({'targetid': train_embeddings['targetid'], 
                             'spectra_features': train_embeddings['spectra_features']})

        test_table  = Table({'targetid': test_embeddings['targetid'], 
                             'spectra_features': test_embeddings['spectra_features']})
        
    provabgs = Table.read('/mnt/home/lparker/ceph/BGS_ANY_full.provabgs.sv3.v0.hdf5')
    provabgs = provabgs[(provabgs['LOG_MSTAR'] > 0) * (provabgs['MAG_G'] > 0) * (provabgs['MAG_R'] > 0) * (provabgs['MAG_Z'] > 0)]
    inds = np.random.permutation(len(provabgs))
    provabgs = provabgs[inds]

    train_provabgs = join(provabgs, train_table, keys_left='TARGETID', keys_right='targetid')
    test_provabgs  = join(provabgs, test_table, keys_left='TARGETID', keys_right='targetid')
    
    return train_provabgs, test_provabgs
# -------------------------------------------- #

def get_data(embedding_file, images=True):
    train_provabgs, test_provabgs = get_provabgs(embedding_file, images)
    
    # Scale the galaxy property data
    prop_scalers = {}
    y_train, y_test = torch.zeros((len(train_provabgs), 5)), torch.zeros((len(test_provabgs), 5))
    for i, p in enumerate(properties):
        prop_train, prop_test = train_provabgs[p].reshape(-1, 1), test_provabgs[p].reshape(-1, 1)
        if p == 'Z_MW': 
            prop_train, prop_test = np.log(prop_train), np.log(prop_test)
        if p == 'SFR': 
            prop_train, prop_test = np.log(prop_train)-train_provabgs['LOG_MSTAR'].reshape(-1, 1), np.log(prop_test)-test_provabgs['LOG_MSTAR'].reshape(-1, 1)
        
        prop_scaler = StandardScaler().fit(prop_train)    
        prop_train, prop_test = prop_scaler.transform(prop_train), prop_scaler.transform(prop_test)
        y_train[:, i], y_test[:, i] = torch.tensor(prop_train.squeeze(), dtype=torch.float32), torch.tensor(prop_test.squeeze(), dtype=torch.float32)
        prop_scalers[p] = prop_scaler
    
    if images:
        train_images, test_images = train_provabgs['image_features'], test_provabgs['image_features']
        image_scaler = StandardScaler().fit(train_images)
        train_images, test_images = image_scaler.transform(train_images), image_scaler.transform(test_images)
    
        data = {'X_train': train_images,
                'X_test': test_images,
                'y_train': y_train,
                'y_test': y_test}
        
    else:
        train_spectra, test_spectra = train_provabgs['spectra_features'], test_provabgs['spectra_features']
        spectrum_scaler = StandardScaler().fit(train_spectra)
        train_spectra, test_spectra = spectrum_scaler.transform(train_spectra), spectrum_scaler.transform(test_spectra)
        
        data = {'X_train': train_spectra,
                'X_test': test_spectra,
                'y_train': y_train,
                'y_test': y_test}
    
    return data, prop_scalers

def main(embedding_file, save_dir, source='images', train_type='zero_shot'):
    if source == 'images': 
        data, prop_scalers = get_data(embedding_file, images=True)
        baseline = torch.load('baseline_models/resnet_results')
        baseline_type = 'ResNet'
    elif source == 'spectra': 
        data, prop_scalers = get_data(embedding_file, images=False)
        baseline = torch.load('baseline_models/spender_results')
        baseline_type = 'Spender'
    else: raise ValueError('Only accepts images or spectra')

    baseline_true, baseline_pred = {}, {}        
    for i, prop in enumerate(properties):
        baseline_true[prop] = baseline['test_trues'][:, i]
        baseline_pred[prop] = baseline['test_preds'][:, i]
    
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    if train_type == 'zero_shot':
        preds = zero_shot_train(data['X_train'], data['y_train'], data['X_test'], data['y_test'], properties, r2_only=False)
    elif train_type == 'few_shot':
        model = SimpleMLP(512, 5, 128, 3)
        preds = few_shot_train(model, data['X_train'], data['y_train'], data['X_test'], data['y_test'], properties, r2_only=False)
    else: raise ValueError('Only accepts zero_shot or few_shot')
    
    truth = data['y_test']
    
    fig, axes = plt.subplots(2, len(properties), figsize=(30, 10))  # 5 subplots, 1 row, 25x5 figure size
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.6)

    for ax, (property, config) in zip(axes[0], prop_dict.items()):
        idx = properties.index(property)
        scaler = prop_scalers[property]

        true_data = scaler.inverse_transform(truth[:, idx].reshape(-1, 1)).squeeze()
        preds_data = scaler.inverse_transform(preds[:, idx].reshape(-1, 1)).squeeze()

        sns.scatterplot(ax=ax, x=true_data, y=preds_data, s=5, color=".15")
        sns.histplot(ax=ax,x=true_data, y=preds_data, bins=64, pthresh=.1, cmap="mako")
        sns.kdeplot(ax=ax,x=true_data, y=preds_data, levels=5, color="w", linewidths=1)

        ax.plot(*config['lim_range'], *config['lim_range'], color='gray', ls='--')
        ax.set_xlim(*config['lim_range'])
        ax.set_ylim(*config['lim_range'])
        ax.text(0.9, 0.1, '$R^2$ score: %0.2f' % r2_score(true_data, preds_data), horizontalalignment='right', verticalalignment='top', fontsize=25, transform=ax.transAxes)
        ax.set_title(property_titles[idx], fontsize=25)
        
    if axes[0].size > 0:
        axes[0][0].set_ylabel("Zero-Shot AstroFM", fontsize=25)  # pad for some spacing
 
    for ax, (property, config) in zip(axes[1], prop_dict.items()):
        idx = properties.index(property)
        true_data = baseline['scalers'][property].inverse_transform(baseline_true[property].reshape(-1, 1)).squeeze()
        preds_data = baseline['scalers'][property].inverse_transform(baseline_pred[property].reshape(-1, 1)).squeeze()

        sns.scatterplot(ax=ax, x=true_data, y=preds_data, s=5, color=".15")
        sns.histplot(ax=ax,x=true_data, y=preds_data, bins=64, pthresh=.1, cmap="mako")
        sns.kdeplot(ax=ax,x=true_data, y=preds_data, levels=5, color="w", linewidths=1)

        ax.plot(*config['lim_range'], *config['lim_range'], color='gray', ls='--')
        ax.set_xlim(*config['lim_range'])
        ax.set_ylim(*config['lim_range'])
        ax.text(0.9, 0.1, '$R^2$ score: %0.2f' % r2_score(true_data, preds_data), horizontalalignment='right', verticalalignment='top', fontsize=25, transform=ax.transAxes)
        ax.set_title(property_titles[idx], fontsize=25)
        
    if axes[1].size > 0:
        axes[1][0].set_ylabel(f'Supervised {baseline_type}', fontsize=25)  # pad for some spacing

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the rect so the main title is visible and not overlapping
    plt.savefig(os.path.join(save_dir, source + f'_scatter_{train_type}.png'))
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Zero- or few- shot learning on images or spectra embeddings")
    parser.add_argument('--embedding_file', type=str, default='/mnt/home/lparker/ceph/property_embeddings.h5', help='Path to the embedding file.')
    parser.add_argument('--save_dir', type=str, help='File to save R2 radar graphs.')
    parser.add_argument('--source', type=str, default='images', help='list spectra or images')
    parser.add_argument('--train_type', type=str, default='zero_shot', help='list zero_shot or few_shot')
    args = parser.parse_args()
    main(args.embedding_file, args.save_dir, source=args.source)
