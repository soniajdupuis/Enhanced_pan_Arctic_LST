from torch.utils.data import DataLoader
from torchgeo.datasets import BoundingBox
from torchgeo.datasets.utils import download_and_extract_archive
from collections.abc import Callable, Iterable
from pathlib import Path
import torch.nn as nn
from torchvision.transforms import Normalize
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential

import numpy as np
import torch
from torch import is_tensor, optim
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import Normalize
from tqdm import tqdm
import csv
import rasterio as rio
from sklearn.metrics import jaccard_score

from torchgeo.datasets import RasterDataset

from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, cast
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential
# Imports from your module
from .utils import downsample, bicubic_with_mask




# Clean fill values and set them to NaN
class CleanFillValues(K.IntensityAugmentationBase2D):
    """Clean fill values (e.g., -32768) from the LST data and set them to NaN."""

    def __init__(self, fill_value: float = -32768) -> None:
        super().__init__(p=1)
        self.fill_value = fill_value

    def apply_transform(self, input: torch.Tensor, params: dict, flags: dict, transform=None) -> torch.Tensor:
        input = input.clone()  # Avoid in-place ops on tensors used downstream
        input[input == self.fill_value] = float('nan')
        return input

# Apply scaling and offset
class ApplyScalingAndOffset(K.IntensityAugmentationBase2D):
    """Apply scale factor and offset to convert raw values to Kelvin."""

    def __init__(self, scale_factor: float, offset: float) -> None:
        super().__init__(p=1)
        self.scale_factor = scale_factor
        self.offset = offset

    def apply_transform(self, input: torch.Tensor, params: dict, flags: dict, transform=None) -> torch.Tensor:
        return input * self.scale_factor + self.offset

# Min-max normalization
class MinMaxScaleLST(K.IntensityAugmentationBase2D):
    """Normalize LST data to the range [0, 1] using min/max values."""

    def __init__(self, min_val: float = 217.65, max_val: float = 338.18) -> None:
        super().__init__(p=1)
        self.min_val = min_val
        self.max_val = max_val

    def apply_transform(self, input: torch.Tensor, params: dict, flags: dict, transform=None) -> torch.Tensor:
        normalized = (input - self.min_val) / (self.max_val - self.min_val + 1e-10)
        return torch.clamp(normalized, 0, 1)


# MinMax normalization for a single channel with clamping
class MinMaxNormalizeChannel(nn.Module):
    """Normalize a single channel to the range [0, 1] using min/max values with clamping."""

    def __init__(self, min_val: float, max_val: float):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized = (input - self.min_val) / (self.max_val - self.min_val + 1e-10)
        return torch.clamp(normalized, 0.0, 1.0)

class FillNaN(nn.Module):
    """Replace NaN values with a specified fill value."""

    def __init__(self, fill_value: float = 0.0):
        super().__init__()
        self.fill_value = fill_value

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(input, nan=self.fill_value)




class CustomRasterDataset(RasterDataset):
    """Custom RasterDataset for temperature data with NaN masks, downsampling, and hole skipping."""

    def __init__(self, *args, scaling:int = 5,split:str = 'train', **kwargs):
        super().__init__(*args, **kwargs)
        
        self.scaling = scaling
        self.split= split

    def __getitem__(self, query: BoundingBox, called: int = 0) -> dict[str, Any]:
        """Retrieve temperature data, masks, and downsampled version."""
        # Get the data from the original RasterDataset (image/mask)
        if called > 32:
            print('called more than 32')
            raise ValueError
        sample = super().__getitem__(query)
        #print(sample) # dict
        data1= sample['image']
        
        
        #if self.transforms:
            #data1 = self.transforms(sample['image'])

        # Extract the temperature data (assuming 'image' contains the temperature data)
        source = downsample(data1, self.scaling).squeeze().unsqueeze(0)
        data1 = data1.squeeze().unsqueeze(0)
        

        # Create NaN mask based on temperature threshold
        # high resolution / low resolution
        mask_hr = (~torch.isnan(data1)).float()
        mask_lr = (~torch.isnan(source)).float()

        data1[mask_hr == 0.] = 0.
        source[mask_lr == 0.] = 0.

        # Skip the sample if there are too many holes (NaNs)
        if  self.split=='train' and (torch.mean(mask_lr) < 0.9 or torch.mean(mask_hr) < 0.8):
            print(f"Too many holes in sample {query}, retrying... ({called + 1} attempt(s))")
            # Skip this sample and recursively try to fetch another
            # Shift the bounding box by a small amount
            shift = 0.05  # degrees, adjust if needed
            new_query = BoundingBox(
                minx=query.minx - shift,
                maxx=query.maxx - shift,
                miny=query.miny + shift,
                maxy=query.maxy + shift,
                mint=query.mint,
                maxt=query.maxt
            )
            return self.__getitem__(new_query, called=called + 1)

        try:
            print(query)
            y_bicubic = torch.from_numpy(bicubic_with_mask(
                source.squeeze().numpy(), mask_lr.squeeze().numpy(), self.scaling)).float()
            y_bicubic = y_bicubic.reshape((1, y_bicubic.shape[0], y_bicubic.shape[1]))


            # Prepare the final sample
            final_sample = {
                'crs': self.crs,
                'bbox': query,
                'lst': data1,  # The original temperature data
                'source': source,  # Downsampled data
                'mask_hr': mask_hr,  # High-resolution mask
                'mask_lr': mask_lr,  # Low-resolution mask
                'y_bicubic': y_bicubic,  # Bicubic downsampled data
            }
            #print(final_sample)
            return final_sample
        except:
            return self.__getitem__(query, called=called + 1)


