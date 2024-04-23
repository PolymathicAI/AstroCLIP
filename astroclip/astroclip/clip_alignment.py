from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity
from tutorial_helpers import load_model_from_ckpt, forward, slice, fnc, dr2_rgb, scatter_plot_as_images
import lightning as L
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
from datasets import load_dataset, load_from_disk
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


sys.path.insert(0, os.path.abspath('/mnt/home/lparker/Documents/AstroFoundationModel/AstroDino/dinov2/dinov2'))
sys.path.insert(0, os.path.abspath('/mnt/home/lparker/Documents/AstroFoundationModel/AstroDino/dinov2'))


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

from torchvision.transforms import CenterCrop, Normalize, Resize, Compose, ToTensor, InterpolationMode
from torchvision import transforms

import wandb
from lightning.pytorch.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import argparse

# Specify where you want the data to be saved locally
CACHE_DIR = '/mnt/ceph/users/lsarra/datasets_astroclip'

MEAN = (0.485, 0.456, 0.406) # Imagenet default mean
STD = (0.229, 0.224, 0.225) # Imagenet default std        
img_transforms = Compose([
    Normalize(MEAN, STD)])

# Custom Dataset to load and shuffle data
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, images, spectra, targetids):
        self.images = images
        self.spectra = spectra
        self.targetids = targetids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.spectra[idx], self.targetids[idx]

class OutputExtractor(L.LightningModule):
    """
    Pass data through network to extract model outputs
    """
    def __init__(self, backbone, freeze_backbone, embed_dim=128, nhead=4, model_embed_dim=1024, dropout=0.1):    
        super().__init__()
        
        self.backbone = backbone
        self.backbone.eval()
        
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim)) 
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, batch_first=True, kdim=model_embed_dim, vdim=model_embed_dim)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        d_model = embed_dim
        dim_feedforward = 4*d_model
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()
        self.freeze_backbone = freeze_backbone
        
    def forward(self, batch, return_weights=False):
        x, _ = batch
        batch_size = x.shape[0]
        
        if self.freeze_backbone:
            with torch.no_grad():
                embedding = self.backbone(x)
        else:
            embedding = self.backbone(x)
                
                    
        attentions = self.multihead_attn(query=self.query.repeat(batch_size,1,1), key=embedding, value=embedding, need_weights=return_weights, average_attn_weights = False)
           
        x = self.norm(self.dropout1(attentions[0]))
        
        #print(x.shape)
                    
        # Small MLP head 
        x = self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))
        return x.squeeze()

class seq_decoder(nn.Module):

    def __init__(self,  model,freeze_backbone, embed_dim=128, nhead=4, model_embed_dim=768, dropout=0.1,
                 ):
        super().__init__()

        # The query of the spectrum transformer is set to a learnable 
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim)) 
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, batch_first=True, kdim=model_embed_dim, vdim=model_embed_dim)
        self.model = model
        
        # Freeze all of the weights of all of the intermediate layers of the pretraned Spectrum Transformer
        for param in self.model.parameters():
            param.requires_grad = False
        self.embed_dim = embed_dim
        d_model = embed_dim
        dim_feedforward = 4 * d_model

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.freeze_backbone = freeze_backbone
        self.activation = nn.GELU()

    def forward(self, sample, return_weights=False):
        sample=sample.unsqueeze(-1)
        batch_size = len(sample)
        
        # Embed the spectrum using the pretrained model 
        if self.freeze_backbone:
            with torch.no_grad():
                embedding = self.model(fnc(sample))['embedding']
        else:
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
    def __init__(self, image_encoder, spectrum_encoder, temperature):
        super().__init__()
        self.image_encoder = image_encoder
        self.spectrum_encoder = spectrum_encoder
        self.temperature= temperature
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
        im, sp, _ = batch
        im = img_transforms(im)
        image_features = self.image_encoder((im.cuda(), None))
        spectrum_features = self.spectrum_encoder(sp)
        loss_withlogit = self.criterion(image_features, spectrum_features, self.temperature)
        self.log("train_loss_withlogit", loss_withlogit)
        loss_nologit = self.criterion(image_features, spectrum_features, 1)
        self.log("train_loss_nologit", loss_nologit)
        self.log("scale", self.logit_scale)
        return loss_withlogit

    def validation_step(self, batch, batch_idx):
        # Validation_step defines the validation loop. It is independent of forward
        im, sp, _ = batch
        im = img_transforms(im)
        image_features = self.image_encoder((im.cuda(), None))
        spectrum_features = self.spectrum_encoder(sp)
        val_loss_nologit = self.criterion(image_features, spectrum_features, 1)
        self.log("val_loss_nologit", val_loss_nologit)
        val_loss_withlogit = self.criterion(image_features, spectrum_features, self.temperature)
        self.log("val_loss_withlogit", val_loss_withlogit)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.05)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=5e-7)
        return {
        'optimizer': optimizer,
        'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'epoch',  # or 'step' for step-wise updating
            'frequency': 1,  # how often to apply
        }
    }

def forward_im(self, x: torch.tensor):
    x = self.patch_embed(x)
    for blk in self.blocks:
            x = blk(x)
    x = self.norm(x)
    return x
    
def generate_embeddings(model, loader, save_path, type="train",device='cuda'):
    model.to(device)
    im_embeddings = []
    sp_embeddings = []
    targetids = []
    with torch.no_grad():
        for batch in tqdm(loader): 
            im = img_transforms(batch[0]).to(device)
            sp = batch[1].squeeze().to(device)
            im_embeddings.append(CLIP(im).detach().cpu().numpy())
            sp_embeddings.append(CLIP(sp, False).detach().cpu().numpy())
            targetids.append(batch[2].detach().cpu().numpy())
    image_features = np.concatenate(im_embeddings)
    spectrum_features = np.concatenate(sp_embeddings)
    targetids = np.concatenate(targetids)

    os.makedirs(save_path, exist_ok=True)            
    with h5py.File(f"{save_path}/file.h5py", 'a') as f:
        train = f.create_group(type)
        train.create_dataset('targetid', data=targetids)
        train.create_dataset('image_features', data=image_features)
        train.create_dataset('spectrum_features', data=spectrum_features)

class config:
    output_dir = '/mnt/home/lparker/ceph/dino_training'
    config_file = '/mnt/home/lparker/Documents/AstroFoundationModel/AstroDino_legacy/astrodino/configs/ssl_default_config.yaml'
    pretrained_weights = '/mnt/home/lparker/ceph/astrodino/vitl12_simplified_better_wd/training_199999/teacher_checkpoint.pth'
    opts = []

if __name__ == '__main__':
    # Set up arg parser for batch size and embedding save directory
    parser = argparse.ArgumentParser(description='Train AstroCLIP')
    parser.add_argument('--embedding_dir', type=str, help='Directory to save embeddings')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--embed_dim', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--wandb_name', type=str, default='astroclip-clip-align', help='Name of the wandb run')
    parser.add_argument('--temperature', type=float, default=15.5, help='Softmax temperature in loss for training')
    parser.add_argument('--freeze_backbone', type=int, default=1, help='Train or freeze pre-trained embedders?')
    args = parser.parse_args()
        
    # Define DINO model
    model, dtype = setup_and_build_model(config())
    
    # Extract encoder_q from Moco_v2 model
    model.forward = forward_im.__get__(model)
    embed_dim = args.embed_dim
    img_model = OutputExtractor(model,embed_dim=embed_dim, freeze_backbone=args.freeze_backbone)
    num_params = np.sum(np.fromiter((p.numel() for p in img_model.parameters()), int))
    print(f"Number of parameters in image model: {num_params:,}")
    
    # The model is saved in the Seqformer branch of Fi-LLM
    model_path = "/mnt/home/sgolkar/ceph/saves/fillm/run-seqformer-2708117"
    out = load_model_from_ckpt(model_path)
    config = out['config']
    spec_model = out['model']
    spec_model.forward = forward.__get__(spec_model, type(img_model))
    num_params = np.sum(np.fromiter((p.numel() for p in spec_model.parameters()), int))
    print(f"Number of parameters in spectrum model: {num_params:,}")

    # Define image and spectrum encoders
    image_encoder = img_model
    spectrum_encoder = seq_decoder(model=spec_model, embed_dim=embed_dim, freeze_backbone=args.freeze_backbone)   
    
    # Set up AstroCLIP
    CLIP = AstroCLIP(image_encoder, spectrum_encoder, args.temperature)
    
    wandb.finish()
    wandb_run = wandb.init(entity="flatiron-scipt",project="astroclip-clip-explore", config={"embed_dim":embed_dim, "model_type":"very_complicated", "batch_size":args.batch_size, "args":vars(args)} )
    wandb_logger = WandbLogger(project="astroclip-clip-explore", id=wandb_run.id, entity="flatiron-scipt")
    lr_monitor = LearningRateMonitor(logging_interval='step') 

    save_path= f'{args.embedding_dir}/{wandb_run.id}'

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
    CLIP.cuda()
        
    # Ok from file
    train_images, train_spectra, train_targetids = torch.load('/mnt/home/lparker/ceph/images_train.pt'), torch.load('/mnt/home/lparker/ceph/spectra_train.pt'), torch.load('/mnt/home/lparker/ceph/targetids_train.pt')
    val_images, val_spectra, val_targetids = torch.load('/mnt/home/lparker/ceph/images_val.pt'), torch.load('/mnt/home/lparker/ceph/spectra_val.pt'), torch.load('/mnt/home/lparker/ceph/targetids_val.pt')

    train_dataset = CustomDataset(train_images, train_spectra, train_targetids)
    val_dataset = CustomDataset(val_images, val_spectra, val_targetids)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=31,
        drop_last= True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1024,
        num_workers=31,
        shuffle=False,
        drop_last= True,
    )

    trainer.fit(CLIP, train_dataloader, val_dataloader)

    print('Done Training!')
        
    # Generate features over the train and embedding sets
    train_features = generate_embeddings(CLIP, train_dataloader, save_path, "train")
    test_features = generate_embeddings(CLIP, val_dataloader, save_path, "test")
