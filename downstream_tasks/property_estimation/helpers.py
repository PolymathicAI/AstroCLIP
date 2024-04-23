import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from astropy.table import Table, join
from astropy.table import Table, join
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

sns.set()
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2})

def zero_shot_train(X_train, y_train, X_test, y_test, properties, r2_only=True):
    neigh = KNeighborsRegressor(weights='distance', n_neighbors=64)
    neigh.fit(X_train, y_train)
    preds = neigh.predict(X_test)
    test_r2s = np.zeros(5)
    for i, prop in enumerate(properties):
        test_r2s[i] = r2_score(y_test[:, i], preds[:, i])
    
    if r2_only: return test_r2s
    else: return preds

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, n_hidden=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(n_hidden):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.activation = nn.ReLU()
        
    def forward(self, x):
        for i in range(self.n_hidden+1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x)

def few_shot_train(X_train, y_train, X_test, y_test, properties, r2_only=True):
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    model = MLP(X_train.shape[1], y_train.shape[1], hidden_dim=32, n_hidden=0)
    model.cuda()
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs.cuda()).detach().cpu()
            preds.append(outputs)
            trues.append(labels)
    
    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()

    test_r2s = np.zeros(len(properties))

    for i, prop in enumerate(properties):
        test_r2s[i] = r2_score(trues[:, i], preds[:, i])

    if r2_only: return test_r2s
    else: return preds

import numpy as np
import matplotlib.pyplot as plt
from math import pi

def plot_radar(data_dict, file_path, title='Galaxy Property Estimation', label_key='labels', fontsize=22):
    # Extract label array
    labels = data_dict[label_key]

    # Format labels as Matplotlib's math text strings
    labels = ['$' + label + '$' for label in labels]

    # Validate data
    num_vars = len(labels)
    for key, array in data_dict.items():
        if key != label_key and len(array) != num_vars:
            raise ValueError(f"All arrays must have the same length as the label array.")

    # Create radar chart
    angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # complete the loop

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    # Plot each array on the radar chart
    for key, array in data_dict.items():
        if key != label_key:
            stats = array.tolist()
            stats += stats[:1]  
            ax.plot(angles, stats, label=key)

    # Add labels with specific fontsize
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=fontsize)  # Explicitly set fontsize for xtick labels

    ax.set_ylim(0, 1)

    # Add legend and title with specific fontsize
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.setp(legend.get_texts(), fontsize=fontsize)  # Explicitly set fontsize for legend

    #plt.title(title, fontsize=fontsize + 2)  # Explicitly set fontsize for title
    plt.savefig(file_path)
    plt.close()
    
def photometry_r2(photo_mlp, eval_mode='from_save'):
    if eval_mode == 'from_save': return photo_mlp
    if eval_mode == 'process': raise ValueError('Not Implemented :(')
    
def resnet_r2(resnet18, properties, eval_mode='from_save'):
    if eval_mode == 'from_save':
        r2s = np.zeros(len(properties))
        for i, prop in enumerate(properties):
            r2s[i] = resnet18['r2'][prop]
        return np.array(r2s)
    else: raise ValueError('Not Implemented :(')
    
def spender_r2(spender, properties, eval_mode='from_save'):
    if eval_mode == 'from_save':
        r2s = np.zeros(len(properties))
        for i, prop in enumerate(properties):
            r2s[i] = spender['r2'][prop]
        return np.array(r2s)
    else: raise ValueError('Not Implemented :(')

def gaussian_kernel(size, sigma):
    """Creates a Gaussian kernel for blurring."""
    grid = np.mgrid[size//2 - size + 1:size//2 + 1, size//2 - size + 1:size//2 + 1]
    gaussian = np.exp(-(grid[0]**2 + grid[1]**2) / (2 * sigma ** 2))
    gaussian /= gaussian.sum()
    gaussian = torch.from_numpy(gaussian).float()
    gaussian = gaussian.unsqueeze(0).unsqueeze(0)
    return gaussian.repeat(3, 1, 1, 1)  # Repeat for each color channel

def resnet_augmentations(images, kernel_size=5):
    """
    Apply ResNet augmentations (random flips and Gaussian blur) to a batch of images.
    
    Args:
    images (torch.Tensor): A batch of images with shape [N, 3, 96, 96].
    kernel_size (int): Size of the Gaussian kernel.
    
    Returns:
    torch.Tensor: Augmented images.
    """
    # Random horizontal and vertical flipping
    flip_h = torch.rand(images.size(0)) > 0.5
    flip_v = torch.rand(images.size(0)) > 0.5
    images[flip_h] = images[flip_h].flip(-1)  # Horizontal flip
    images[flip_v] = images[flip_v].flip(-2)  # Vertical flip

    # Apply Gaussian blur with random sigma for each image
    blurred_images = images.clone()
    for i in range(images.size(0)):
        sigma = torch.rand(1).item() * 2  # Random sigma between 0 and 2
        blur_kernel = gaussian_kernel(kernel_size, sigma)
        padding = kernel_size // 2
        blurred_images[i] = F.conv2d(images[i].unsqueeze(0), blur_kernel, padding=padding, groups=3).squeeze(0)

    return blurred_images


