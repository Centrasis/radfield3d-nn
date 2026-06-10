import numpy as np
from typing import Union
from torch import Tensor
import torch
import torch.nn.functional as F
import torch.cuda


def resample_histogram_means(histogram: Union[np.ndarray, Tensor], new_bins: Union[np.ndarray, Tensor], repeat_previous_mean_on_empty_bin: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Resamples a histogram to match new bin widths and calculates the mean y-value for each new bin.

    Parameters:
    - histogram: The histogram to resample.
    - new_bins: The new bin edges for resampling.
    - repeat_previous_mean_on_empty_bin: If True, the mean value of the previous bin will be repeated if a new bin is empty. If False, the bin will be set to nan. Default is False.

    Returns:
    - A tuple containing the new bin edges and the mean y-values for each new bin.
    """
    # Initialize an array to store the mean values for the new bins
    assert isinstance(histogram, (np.ndarray, Tensor)), "histogram must be a numpy array or a PyTorch tensor"
    use_numpy = isinstance(histogram, np.ndarray)

    zeros_fn = np.zeros if use_numpy else torch.zeros
    where_fn = np.where if use_numpy else torch.where
    mean_fn = np.mean if use_numpy else torch.mean
    nan_val = np.nan if use_numpy else torch.nan

    mean_values = zeros_fn((len(new_bins), 2), dtype=histogram.dtype)

    # Iterate over the new bins
    for i in range(1, len(new_bins)):
        # Find the indices of x_values that fall into the current new bin
        indices = where_fn((histogram[:, 0] >= new_bins[i-1]) & (histogram[:, 0] < new_bins[i]))[0]
        
        # Calculate the mean y-value for these indices
        if len(indices) > 0:
            mean_values[i-1][1] = mean_fn(histogram[:, 1][indices])
            mean_values[i-1][0] = new_bins[i-1]
        else:
            mean_values[i-1][1] = (mean_values[i-2][1] if i > 1 else histogram[0, 1]) if repeat_previous_mean_on_empty_bin else nan_val
            mean_values[i-1][0] = new_bins[i-1]

    return mean_values


def resample_histogram_bilinear(histogram: torch.Tensor, target_bins: int) -> torch.Tensor:
    """
    Resample a batch of normalized histograms to a new bin count.
    
    Args:
        histogram: Input histogram tensor of shape (N, M) where N is batch size and M is bin count
        target_bins: The desired number of output bins (M2)
        
    Returns:
        Resampled histogram of shape (N, target_bins)
    """
    batch_size, source_bins = histogram.shape
    
    x_coords = torch.linspace(-1, 1, target_bins, device=histogram.device, dtype=histogram.dtype)

    grid = torch.zeros((batch_size, 1, target_bins, 2), device=histogram.device, dtype=histogram.dtype)
    grid[:, :, :, 0] = x_coords.view(1, 1, -1)
    histogram_input = histogram.view(batch_size, 1, 1, source_bins)

    try:
        resampled = F.grid_sample(histogram_input, grid, mode='bilinear', align_corners=True)
    except RuntimeError as e:
        if "CUDNN" in str(e):
            raise torch.cuda.OutOfMemoryError("Out of memory error during grid_sample. Try reducing the batch size.")
        else:
            raise e
    resampled = resampled.squeeze(1).squeeze(1)
    
    resampled_sum = resampled.sum(dim=1, keepdim=True)
    resampled_sum = torch.where(resampled_sum == 0, torch.ones_like(resampled_sum), resampled_sum)
   
    resampled = resampled / resampled_sum
    
    return resampled
