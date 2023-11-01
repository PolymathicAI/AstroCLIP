from fillm.run.model import *
import torch.nn.functional as F
import torch
import numpy as np
import h5py
from PIL import Image as im
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.stats import binned_statistic_2d

def load_model_from_ckpt(ckpt_path: str):
    """
    Load a model from a checkpoint.
    """
    if Path(ckpt_path).is_dir():
        ckpt_path = Path(ckpt_path) / "ckpt.pt"

    chkpt = torch.load(ckpt_path)
    config = chkpt["config"]
    state_dict = chkpt["model"]
    model_name = config["model"]['kind']
    model_keys = get_model_keys(model_name)
    
    model_args = {k: config['model'][k] for k in model_keys}

    model_ctr, config_cls = model_registry[model_name]
    model_config = config_cls(**model_args)
    model_ = model_ctr(model_config)
    model_.load_state_dict(state_dict)

    return {"model": model_, "config": config}

def forward(
    self, x: torch.Tensor, y: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = x.device
    t = x.shape[1]

    # find the mask locations
    locs = x != y

    if t > self.config.block_size:
        raise ValueError(
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.config.block_size}"
        )
    pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

    # forward the GPT model itself
    data_emb = self.data_embed(x)  # to shape (b, t, embedding_dim)
    pos_emb = self.position_embed(pos)  # to shape (t, embedding_dim)

    x = self.dropout(data_emb + pos_emb)
    embeddings = []
    for block in self.blocks:
        x = block(x)
        embeddings.append(x.detach().clone())
    x = self.final_layernorm(x)

    preds = self.head(x)
    if y is not None:
        # if we are given some desired targets also calculate the loss
        locs = locs.type_as(preds)
        loss = F.mse_loss(preds * locs, y * locs, reduction="mean") / locs.mean()
    else:
        loss = None

    return {"preds": preds, "loss": loss, "embeddings": embeddings}

def slice(x, section_length=10, overlap=5):

    start_indices = np.arange(0, x.shape[1] - overlap, section_length - overlap)
    sections = [x[:,start:start + section_length].transpose(1,2) for start in start_indices]

    # If the last section is not of length 'section_length', you can decide whether to keep or discard it
    if sections[-1].shape[1] < section_length:
        sections.pop(-1)  # Discard the last section  

    return torch.cat(sections, 1)


def fnc(x):
    std, mean = x.std(1, keepdim=True).clip_(0.2), x.mean(1, keepdim=True)
    x = (x - mean) / std
    x = slice(x, 20, 10)
    x = F.pad(x, pad=(2, 0, 1, 0), mode='constant', value=0)
    x[:,0,0] = (mean.squeeze()-2)/2
    x[:,0,1] = (std.squeeze()-2)/8     

    return x

def sdss_rgb(imgs, bands, scales=None,
             m = 0.02):
    import numpy as np
    rgbscales = {'u': (2,1.5), #1.0,
                 'g': (2,2.5),
                 'r': (1,1.5),
                 'i': (0,1.0),
                 'z': (0,0.4), #0.3
                 }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
        
    # b,g,r = [rimg * rgbscales[b] for rimg,b in zip(imgs, bands)]
    # r = np.maximum(0, r + m)
    # g = np.maximum(0, g + m)
    # b = np.maximum(0, b + m)
    # I = (r+g+b)/3.
    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I

    # R = fI * r / I
    # G = fI * g / I
    # B = fI * b / I
    # # maxrgb = reduce(np.maximum, [R,G,B])
    # # J = (maxrgb > 1.)
    # # R[J] = R[J]/maxrgb[J]
    # # G[J] = G[J]/maxrgb[J]
    # # B[J] = B[J]/maxrgb[J]
    # rgb = np.dstack((R,G,B))
    rgb = np.clip(rgb, 0, 1)
    return rgb

def dr2_rgb(rimgs, bands, **ignored):
    return sdss_rgb(rimgs, bands, scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)

# Code borrowed from https://github.com/georgestein/ssl-legacysurvey
def scatter_plot_as_images(z_emb, images, nx=8, ny=8, npix_show=96, iseed=13579, display_image=True):
    """Sample points from scatter plot and display as their original galaxy image
        
    Parameters
    ----------
    DDL : class instance
        DecalsDataLoader class instance
    z_emb: array
        (N, 2) array of the galaxies location in some compressed space. 
        If second axis has a dimensionality greater than 2 we only consider the leading two components.
    """
    z_emb = z_emb[:, :2] # keep only first two dimensions

    nplt = nx*ny

    img_full = np.zeros((ny*npix_show, nx*npix_show, 3)) + 255#, dtype=np.uint8) + 255

    xmin = z_emb[:,0].min()
    xmax = z_emb[:,0].max()
    ymin = z_emb[:,1].min()
    ymax = z_emb[:,1].max()

    dz_emb = 0.25
    dx_cent = z_emb[:,0].mean()
    dy_cent = z_emb[:,1].mean()

    dx_cent = 10.0
    dy_cent = 7.0

    # xmin = dx_cent - dz_emb
    # xmax = dx_cent + dz_emb
    # ymin = dy_cent - dz_emb
    # ymax = dy_cent + dz_emb

    binx = np.linspace(xmin,xmax, nx+1)
    biny = np.linspace(ymin,ymax, ny+1)

    ret = binned_statistic_2d(z_emb[:,0], z_emb[:,1], z_emb[:,1], 'count', bins=[binx, biny], expand_binnumbers=True)
    z_emb_bins = ret.binnumber.T

    inds_used = []
    inds_lin = np.arange(z_emb.shape[0])

    # First get all indexes that will be used
    for ix in range(nx):
        for iy in range(ny):
            dm = (z_emb_bins[:,0]==ix) & (z_emb_bins[:,1]==iy)
            inds = inds_lin[dm]

            np.random.seed(ix*nx+iy+iseed)
            if len(inds) > 0:
                ind_plt = np.random.choice(inds)
                inds_used.append(ind_plt)# inds_use[ind_plt])

    # load in all images
    iimg = 0
    
    # Add each image as postage stamp in desired region  
    for ix in range(nx):
        for iy in range(ny):
            dm = (z_emb_bins[:,0] == ix) & (z_emb_bins[:,1]==iy)
            inds = inds_lin[dm]

            np.random.seed(ix*nx+iy+iseed)
            if len(inds) > 0:

                imi = images[inds[0]][28:-28, 28:-28]
                img_full[iy*npix_show:(iy+1)*npix_show, ix*npix_show:(ix+1)*npix_show] = imi

                iimg += 1
                
    if display_image:
        plt.figure(figsize=(nx, ny))
        plt.imshow(img_full, origin='lower')#, interpolation='none')
        plt.axis('off')
        
    return img_full