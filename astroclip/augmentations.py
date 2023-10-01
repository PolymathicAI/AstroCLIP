import numpy as np
import torch 
import torch.nn.functional as F

class ToRGB:
    '''takes in batched image of size (batch_size, npix, npix, nchannel), 
    and converts from native telescope image scaling to rgb'''

    def __init__(self, scales=None, m=0.03, Q=20., bands=['g', 'r', 'z']):
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
        
        # rearrange scales to correspond to image channels after swapping
        self.scales = [self.scales[i] for i in self.axes]

    def __call__(self, image):
        # Assuming image is of shape [batch_size, npix, npix, nchannel]
        image = image.permute(0, 3, 1, 2)  # Change to [batch_size, nchannel, npix, npix]
        image = image[:, self.axes]
        image = image.permute(0, 2, 3, 1)  # Change back to [batch_size, npix, npix, nchannel]
        scales = torch.tensor(self.scales, dtype=torch.float32).to(image.device)

        I = torch.sum(torch.clamp(image * scales + self.m, min=0), dim=-1) / len(self.bands)
        
        fI = torch.arcsinh(self.Q * I) / np.sqrt(self.Q)
        I += (I == 0.) * 1e-6
        
        image = (image * scales + self.m) * (fI / I).unsqueeze(-1)
        image = torch.clamp(image, 0, 1)

        return image.permute(0, 3, 1, 2)  # Change to [batch_size, nchannel, npix, npix]

class AddGaussianNoise(object):
    def __init__(self, mean=0., std_max=1.):
        self.std_max = std_max
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + (torch.randn(tensor.size()) * torch.rand(tensor.shape[0]).reshape([-1,1,1,1]) * self.std_max + self.mean).to(tensor.device)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

class SpectrumNoising():

    def __init__(self, 
                 residual_std, 
                 smoothing_kernel_std=5.):
        self.residual_std = torch.from_numpy(residual_std).to('cuda')
        self.smoothing_kernel_std = smoothing_kernel_std
        self.kernel = self.gaussian_filter_1d(int(smoothing_kernel_std)*3, smoothing_kernel_std).view(1, 1, -1).to('cuda')

    def gaussian_filter_1d(self, size: int, sigma: float) -> torch.Tensor:
        """
        Create a 1D Gaussian filter tensor.
        
        Args:
        - size (int): The size of the filter. It should be an odd number.
        - sigma (float): The standard deviation of the Gaussian.
        
        Returns:
        - torch.Tensor: A 1D Gaussian filter tensor.
        """
        assert size % 2 == 1, "Size should be an odd number."
        
        # Create a tensor of size 'size' with values from -size//2 to size//2
        x = torch.arange(-size // 2, size // 2 + 1, dtype=torch.float32)
        
        # Compute the 1D Gaussian filter
        gaussian = torch.exp(-x**2 / (2 * sigma**2))
        gaussian /= gaussian.sum()
        return gaussian
    
    @torch.no_grad()
    def __call__(self, spectrum):
        """
        Apply a 1D Gaussian filter to a tensor.
        
        Args:
        - input_tensor (torch.Tensor): The input tensor of shape (batch_size, channels, length).
        - filter_tensor (torch.Tensor): The 1D Gaussian filter tensor.
        
        Returns:
        - torch.Tensor: The filtered tensor.
        """
        spectrum = spectrum.squeeze()
        
        # Apply the convolution
        spectrum_smooth = F.conv1d(spectrum.unsqueeze(1), self.kernel, padding=self.kernel.shape[-1] // 2).squeeze()
        
        # Generate fake noise
        noise = torch.randn_like(spectrum) * self.residual_std.expand_as(spectrum)

        # Fake spectrum which is a little bit smoother and with random noise
        fake_spectrum = spectrum_smooth + noise

        # Building a linear combination of both versions
        l = torch.rand(1)

        return l * spectrum + (1 - l) * fake_spectrum
