from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity
from tutorial_helpers import load_model_from_ckpt, forward, slice, fnc, dr2_rgb, scatter_plot_as_images
import lightning as L
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
from datasets import load_dataset
from torchvision.transforms import Compose, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, RandomErasing, ToTensor, CenterCrop, ToPILImage
from fillm.run.model import *
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import numpy as np
import warnings
import seaborn as sns
import torch.optim.lr_scheduler as lr_scheduler
import os
import sys

sys.path.insert(0, os.path.abspath('/mnt/home/lparker/Documents/AstroFoundationModel/AstroDino/dinov2/'))

from dinov2.utils.config import setup
from dinov2.models import build_model_from_cfg
from dinov2.fsdp import FSDPCheckpointer
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.eval.setup import setup_and_build_model
from astropy.io import fits
from dinov2.data.transforms import (
    make_normalize_transform,
)

import numpy as np
import h5py
from PIL import Image as im
from tqdm import tqdm

from torchvision.transforms import CenterCrop, Normalize
from torchvision import transforms

import wandb
from lightning.pytorch.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import argparse

MEAN = (0.485, 0.456, 0.406) # Imagenet default mean
STD = (0.229, 0.224, 0.225) # Imagenet default std

def make_normalize_transform(mean = MEAN, std  = STD) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)

crop = CenterCrop(144)
normalize = make_normalize_transform(mean=MEAN, std=STD)

# Specify where you want the data to be saved locally
CACHE_DIR = '/mnt/ceph/users/lparker/datasets_astroclip'

def sdss_rgb(imgs, bands, scales=None,
             m = 0.02):
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
        img = torch.maximum(torch.tensor(0), img * scale + m)
        I = I + img
    I /= len(bands)
    Q = 20
    fI = torch.arcsinh(Q * I) / torch.sqrt(torch.tensor(Q))
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = torch.zeros((H,W,3)).to(torch.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I
    rgb = torch.clip(rgb, 0, 1)
    return rgb

def dr2_rgb(rimgs, bands, **ignored):
    return sdss_rgb(rimgs, bands, scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)), m=0.03)

class toRGB(transforms.ToTensor):
    def __init__(self, bands, scales=None, m=0.02):
        self.bands = bands
        self.scales = scales
        self.m = m

    def __call__(self, rimgs):
        if len(rimgs.shape) == 3:
            return dr2_rgb(rimgs.T, self.bands).T
        if len(rimgs.shape) == 4:
            img_outs = []
            for img in rimgs:
                img_outs.append(dr2_rgb(img.T, self.bands).T[None, :, :, :])
            return torch.concatenate(img_outs)
        
def forward_im(self, x: torch.tensor):
    x = self.patch_embed(x)
    for blk in self.blocks:
            x = blk(x)
    x = self.norm(x)
    return x

class OutputExtractor(L.LightningModule):
    """
    Pass data through network to extract model outputs
    """
    def __init__(self, backbone, embed_dim=128, nhead=4, model_embed_dim=1024, dropout=0.1):    
        super().__init__()
        
        self.backbone = backbone
        self.backbone.eval()
        
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim)) 
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, batch_first=True, kdim=model_embed_dim, vdim=model_embed_dim)
        
        #for param in self.backbone.parameters():
        #    param.requires_grad = False
            
        d_model = embed_dim
        dim_feedforward = 4*d_model
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, batch, return_weights=False):
        x, _ = batch
        batch_size = x.shape[0]
        
        with torch.no_grad():
            embedding = self.backbone(x)
                    
        attentions = self.multihead_attn(query=self.query.repeat(batch_size,1,1), key=embedding, value=embedding, need_weights=return_weights, average_attn_weights = False)
           
        x = self.norm(self.dropout1(attentions[0]))
        
        #print(x.shape)
                    
        # Small MLP head 
        x = self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))
        return x.squeeze()

class seq_decoder(nn.Module):

    def __init__(self,  model, embed_dim=128, nhead=4, model_embed_dim=768, dropout=0.1):
        super().__init__()

        # The query of the spectrum transformer is set to a learnable 
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim)) 
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, batch_first=True, kdim=model_embed_dim, vdim=model_embed_dim)
        self.model = model
        
        # Freeze all of the weights of all of the intermediate layers of the pretraned Spectrum Transformer
        #for param in self.model.parameters():
        #    param.requires_grad = False

        d_model = embed_dim
        dim_feedforward = 4 * d_model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, sample, return_weights=False):
        sample=sample.unsqueeze(-1)
        batch_size = len(sample)
        
        # Embed the spectrum using the pretrained model 
        with torch.no_grad():
            embedding = self.model(fnc(sample))['embedding']
            
        # Perform single cross-attention on the embeddings
        attentions = self.multihead_attn(query=self.query.repeat(batch_size,1,1), key=embedding, value=embedding, need_weights=return_weights, average_attn_weights = False)
        x = self.norm(self.dropout1(attentions[0]))
        
        # Small MLP head 
        x = x+self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        
        # Return weights option (for attention mask)
        if return_weights:
            return x.squeeze(), attentions[1]
        
        return x.squeeze()
    
class CLIPLoss(nn.Module):
    def get_logits(self, image_features, spectrum_features, logit_scale):
        image_features = F.normalize(image_features, dim=-1, eps=1e-3)
        spectrum_features = F.normalize(spectrum_features, dim=-1, eps=1e-3)
        logits_per_image = logit_scale * image_features @ spectrum_features.T
        return logits_per_image, logits_per_image.T

    def forward(self, image_features, spectrum_features, logit_scale, output_dict=False):
        logits_per_image, logits_per_spectrum = self.get_logits(image_features, spectrum_features, logit_scale)
        labels = torch.arange(logits_per_image.shape[0], device=image_features.device, dtype=torch.long)
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_spectrum, labels)
        ) / 2
        return {"contrastive_loss": total_loss} if output_dict else total_loss
    
class AstroCLIP(L.LightningModule):
    def __init__(self, image_encoder, spectrum_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.spectrum_encoder = spectrum_encoder
        
        # Logit scale is fixed to 15.5 and is not a learnable parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(15.5))
        self.criterion = CLIPLoss()
        
    def forward(self, x, image=True, return_weights=False):
        if image:
            # Embed image
            embedding = self.image_encoder((x,None))
        else:
            # Embed spectrum
            embedding = self.spectrum_encoder(x, return_weights=return_weights)
        return embedding


    def training_step(self, batch, batch_idx):
        # Training_step defines the train loop. It is independent of forward
        im, sp = batch['image'], batch['spectrum'].squeeze()
        im = image_transforms(im).to('cuda')
        image_features = self.image_encoder((im, None))
        spectrum_features = self.spectrum_encoder(sp)
        loss_withlogit = self.criterion(image_features, spectrum_features, 15.5)
        self.log("train_loss_withlogit", loss_withlogit)
        loss_nologit = self.criterion(image_features, spectrum_features, 1)
        self.log("train_loss_nologit", loss_nologit)
        self.log("scale", self.logit_scale)
        return loss_withlogit

    def validation_step(self, batch, batch_idx):
        # Validation_step defines the validation loop. It is independent of forward
        im, sp = batch['image'], batch['spectrum'].squeeze()
        im = image_transforms(im).to('cuda')
        image_features = self.image_encoder((im, None))
        spectrum_features = self.spectrum_encoder(sp)
        val_loss_nologit = self.criterion(image_features, spectrum_features, 1)
        self.log("val_loss_nologit", val_loss_nologit)
        val_loss_withlogit = self.criterion(image_features, spectrum_features, 15.5)
        self.log("val_loss_withlogit", val_loss_withlogit)
    
    def configure_optimizers(self):
        #optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.1)
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5, weight_decay=0.01)
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=5e-6)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=5e-7)
        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'epoch',  # or 'step' for step-wise updating
            'frequency': 1,  # how often to apply
        }
    }
    
def generate_embeddings(model, loader, device='cuda'):
    model.to(device)
    im_embeddings = []
    sp_embeddings = []
    targetids = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader): 
            im = image_transforms(batch['image']).to(device)
            sp = batch['spectrum'].squeeze().to(device)
            im_embeddings.append(CLIP(im).detach().cpu().numpy())
            sp_embeddings.append(CLIP(sp, False).detach().cpu().numpy())
            targetids.append(batch['targetid'].detach().cpu().numpy())
    image_features = np.concatenate(im_embeddings)
    spectrum_features = np.concatenate(sp_embeddings)
    targetids = np.concatenate(targetids)

    image_features_normed = image_features / np.linalg.norm(image_features, axis=-1, keepdims=True)
    spectrum_features_normed = spectrum_features / np.linalg.norm(spectrum_features, axis=-1, keepdims=True)

    return {'image_features': image_features_normed, 
            'spectrum_features': spectrum_features_normed, 
            'targetid': targetids}

class config:
    output_dir = '/mnt/home/lparker/ceph/dino_training'
    config_file = '/mnt/home/lparker/Documents/AstroFoundationModel/AstroDino_legacy/astrodino/configs/ssl_default_config.yaml'
    pretrained_weights = '/mnt/home/lparker/ceph/astrodino/vitl12_simplified_better_wd/training_199999/teacher_checkpoint.pth'
    opts = []

if __name__ == '__main__':
    # Set up arg parser for batch size and embedding save directory
    parser = argparse.ArgumentParser(description='Train AstroCLIP')
    parser.add_argument('--embedding_file', type=str, help='Directory to save embeddings')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--wandb_name', type=str, default='astroclip-clip-align', help='Name of the wandb run')
    args = parser.parse_args()
    
    # Load the dataset from Huggingface
    dataset = load_dataset('/mnt/home/lparker/Documents/AstroFoundationModel/AstroCLIP/astroclip_datasets/legacy_survey.py', cache_dir=CACHE_DIR)
    dataset.set_format(type='torch', columns=['image', 'spectrum', 'targetid'])

    # Create the dataloaders
    train_dataloader = torch.utils.data.DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True, num_workers=10)
    val_dataloader = torch.utils.data.DataLoader(dataset['test'], batch_size=args.batch_size, shuffle=False, num_workers=10)

    # Define Transforms to be used during training
    image_transforms = Compose([
            toRGB(bands=['g', 'r', 'z']),
            RandomVerticalFlip(),
            RandomHorizontalFlip(),    
            CenterCrop(144),
            Normalize(mean=MEAN, std=STD)
    ])
    
    # Define DINO model
    model, dtype = setup_and_build_model(config())
    
    # Extract encoder_q from Moco_v2 model
    model.forward = forward_im.__get__(model)
    img_model = OutputExtractor(model)
    
    # The model is saved in the Seqformer branch of Fi-LLM
    model_path = "/mnt/home/sgolkar/ceph/saves/fillm/run-seqformer-2708117"
    out = load_model_from_ckpt(model_path)

    config = out['config']
    spec_model = out['model']
    
    # Modify the forward to output all of the embeddings
    spec_model.forward = forward.__get__(spec_model, type(img_model))
    
    # Define image and spectrum encoders and the AstroCLIP model
    image_encoder = img_model
    spectrum_encoder = seq_decoder(model=spec_model)   
    CLIP = AstroCLIP(image_encoder, spectrum_encoder)
    
    # Set up WANDB
    wandb_logger = WandbLogger(project="astroclip-clip-align", entity="flatiron-scipt", name="dino_run_logit_attblock")
    lr_monitor = LearningRateMonitor(logging_interval='step') 

    # Define Trainer
    trainer = L.Trainer(
            max_epochs=args.epochs,
            logger=wandb_logger,
            callbacks=[
                lr_monitor,
                ModelCheckpoint(
                    every_n_epochs=1,
                    save_top_k=2,
                    monitor="val_loss_nologit",
                )],
                )
    
    # Train the model
    torch.set_float32_matmul_precision('medium')
    trainer.fit(model=CLIP, 
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader)
        
    # Generate features over the train and embedding sets
    train_features = generate_embeddings(CLIP, train_dataloader)
    test_features = generate_embeddings(CLIP, val_dataloader)
    
    if not os.path.exists(args.embedding_file):
        os.makedirs(args.embedding_file)
        
    with h5py.File(args.embedding_file, 'w') as f:
        train = f.create_group('train')
        test = f.create_group('test')
        train.create_dataset('targetid', data=train_features['targetid'])
        train.create_dataset('image_features', data=train_features['image_features'])
        train.create_dataset('spectrum_features', data=train_features['spectrum_features'])
        test.create_dataset('targetid', data=test_features['targetid'])
        test.create_dataset('image_features', data=test_features['image_features'])
        test.create_dataset('spectrum_features', data=test_features['spectrum_features'])


    
