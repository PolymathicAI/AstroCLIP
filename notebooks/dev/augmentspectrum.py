import torch


class AugmentSpectrum(object):
    def __init__(self, max_variance, max_length):
        self.max_variance = max_variance
        self.max_length = max_length
        
    def mask_batch_tensor(self, batch_tensor, mask_length):
        batch_size, sequence_length = batch_tensor.shape
        if mask_length <= 0 or mask_length >= sequence_length:
            raise ValueError("Mask length should be greater than 0 and less than length of tensor")

        start_idxs = torch.randint(0, sequence_length - mask_length + 1, (batch_size,)).cuda()
        end_idxs = (start_idxs + mask_length).cuda()

        start_values = torch.gather(batch_tensor, 1, start_idxs.unsqueeze(1)).squeeze(1)
        end_values = torch.gather(batch_tensor, 1, (end_idxs - 1).unsqueeze(1)).squeeze(1)
        steps = (end_values - start_values) / (mask_length - 1)

        masks = torch.ones(batch_size, sequence_length)
        copy_tensor = torch.clone(batch_tensor)
        for i in range(mask_length):
            masks[torch.arange(batch_size), start_idxs + i] = 0
            copy_tensor[torch.arange(batch_size), start_idxs + i] = start_values + i * steps

        return copy_tensor
    
    def batched_convolution(self, batch_tensor, filters):
        batch_size, sequence_length = batch_tensor.shape
        _, filter_size = filters.shape

        # Extending dimensions for broadcasting
        extended_batch = batch_tensor.unsqueeze(2)  # (B, L, 1)
        extended_filters = filters.unsqueeze(1)  # (B, 1, F)

        # Apply each filter
        convolution = extended_batch[:, :sequence_length-filter_size+1, :] * extended_filters  # (B, L-F+1, F)
        convolution = convolution.sum(2)  # Sum over the filter size dimension

        # Padding to keep the sequence_length the same after convolution
        padding = (filter_size - 1) // 2
        convolution = F.pad(convolution, (padding, padding))

        return convolution

    def gaussian_smooth(self, batch_tensor, max_variance, kernel_size=5):
        batch_size, _ = batch_tensor.shape
        variances = torch.rand(batch_size) * max_variance

        half_kernel_size = kernel_size // 2
        x = torch.arange(-half_kernel_size, half_kernel_size + 1).float().to(batch_tensor.device)
        kernels = torch.exp(-x**2 / (2 * variances.view(-1, 1)))
        kernels /= kernels.sum(dim=1, keepdim=True)  # Normalize the kernel
        
        result = self.batched_convolution(batch_tensor, kernels)
        return result
    
    def __call__(self, spectrum):
        smoothed = self.gaussian_smooth(spectrum, self.max_variance).cuda()      
        masked = self.mask_batch_tensor(smoothed, self.max_length)

        return masked