import numpy as np
import skimage.transform
import skimage.filters
import torch
from typing import List

from torchvision import transforms


class RandomGaussianBlur(transforms.RandomApply):
    def __init__(self, *, p: float = 0.5):
        keep_p = 1 - p
        transform = GaussianBlur()
        super().__init__([transform], p=keep_p)


class RandomGaussianNoise(transforms.RandomApply):
    def __init__(self, *, im_dim = 144, p: float = 0.5):
        keep_p = 1 - p
        transform = GaussianNoise(im_dim=im_dim)
        super().__init__([transform], p=keep_p)


class ToRGB:
    """
    Transformation from raw image data (nanomaggies) to the rgb values displayed
    at the legacy viewer https://www.legacysurvey.org/viewer

    Code copied from
    https://github.com/legacysurvey/imagine/blob/master/map/views.py
    """
    def __init__(self, scales=None, m=0.03, Q=20, bands=['g', 'r', 'z']):
        rgb_scales = {'u': (2,1.5),
                      'g': (2,6.0),
                      'r': (1,3.4),
                      'i': (0,1.0),
                      'z': (0,2.2)}
        if scales is not None:
            rgb_scales.update(scales)
            
        self.rgb_scales = rgb_scales
        self.m = m
        self.Q = Q
        self.bands = bands
        self.axes, self.scales = zip(*[rgb_scales[bands[i]] for i in range(len(bands))])
        
        # rearange scales to correspond to image channels after swapping
        self.scales = [self.scales[i] for i in self.axes]

    def __call__(self, imgs):
        # Check image shape and set to C x H x W
        if imgs.shape[0] != len(self.bands):
            imgs = imgs.transpose(1,2,0)
        
        I = 0
        for img,band in zip(imgs, self.bands):
            plane,scale = self.rgb_scales[band]
            img = np.maximum(0, img * scale + self.m)
            I = I + img
        I /= len(self.bands)
            
        Q = 20
        fI = np.arcsinh(Q * I) / np.sqrt(Q)
        I += (I == 0.) * 1e-6
        H,W = I.shape
        rgb = np.zeros((H,W,3), np.float32)
        for img,band in zip(imgs, self.bands):
            plane,scale = self.rgb_scales[band]
            rgb[:,:,plane] = (img * scale + self.m) * fI / I
        
        rgb = np.clip(rgb, 0, 1)
        return rgb


class GaussianNoise:
    """
    Augmentations tuned to the Legacy Survey Data (with minor modifications).

    Code copied from
    https://github.com/georgestein/ssl-legacysurvey/blob/main/ssl_legacysurvey/data_loaders/decals_augmentations.py#L296
    """
    def __init__(self, scaling: List = [1.], mean: float = 0, im_dim: int = 144, im_ch: int = 3, decals: bool = True, uniform: bool = False):
        self.mean = mean
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.2264926, 0.2431146, 0.1334844])
        self.loc_dist  = np.array([-0.0006735, -0.0023663, -0.0143416])
        self.scale_dist  = np.array([0.0037602, 0.0067417, 0.0260779])
        
        self.sigma_dist  = np.log(self.scale_dist)
    
        # noise in channels is uncorrelated, as images taken at dirrerent times/telescopes
        self.noise_ch_min = np.array([0.001094, 0.001094, 0.001094])
        self.noise_ch_max = np.array([0.013, 0.018, 0.061])

    def __call__(self, image: np.ndarray):
        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
    
        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.noise_ch_min, self.noise_ch_max)
        else:
            self.sigma_final = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.] = 0.
        self.sigma_augment = np.sqrt(self.sigma_augment)
        
        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.:
                image[i,:,:] += np.random.normal(self.mean, self.sigma_augment[i], size = (self.im_dim, self.im_dim))

        return image


class GaussianBlur:
    """
    Augmentations tuned to the Legacy Survey Data (with minor modifications).

    Code copied from
    https://github.com/georgestein/ssl-legacysurvey/blob/main/ssl_legacysurvey/data_loaders/decals_augmentations.py#L296
    """
    def __init__(self, scaling: List = [1.], im_dim: int = 144, im_ch: int = 3, decals: bool = True, uniform: bool = False):
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform
        
        # Log normal fit paramaters
        self.shape_dist = np.array([0.2109966, 0.3008485, 0.3471172])
        self.loc_dist  = np.array([1.0807153, 1.2394326, 1.1928363])
        self.scale_dist  = np.array([1.3153171, 0.9164757, 0.8233702])
        
        self.sigma_dist  = np.log(self.scale_dist)

        self.psf_ch_min = np.array([1.3233109, 1.2667341, 1.2126263])
        self.psf_ch_max = np.array([5., 4.5, 4.25])
    

    def __call__(self, image: np.ndarray):
        # noise in channels is uncorrelated, as images taken at different times/telescopes
        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
    
        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.psf_ch_min, self.psf_ch_max)
        else:
            self.sigma_final = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.] = 0.
        self.sigma_augment = np.sqrt(self.sigma_augment)
        
        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.:
                image[i,:,:] = skimage.filters.gaussian(image[i,:,:], sigma=self.sigma_augment[i], mode='reflect')

        return image