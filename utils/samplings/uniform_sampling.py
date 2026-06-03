import numpy as np
from typing import Union
from torch import Tensor


def resample_equidistant(data: Union[np.ndarray, Tensor], max_size: int) -> Union[np.ndarray, Tensor]:
    """
    Resample data to have a maximum size of max_size by selecting equidistant points.
    :param data: Data to be resampled.
    :param max_size: Maximum size of the resampled data.
    :return: Resampled data.
    """
    indices = np.arange(0, len(data) - 1, len(data) / max_size, dtype=float)
    indices = np.round(indices).astype(int)
    indices = np.unique(indices)
    return data[indices]
