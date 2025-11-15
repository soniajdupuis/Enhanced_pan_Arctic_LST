import os
import argparse
import rasterio
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
import time
import numpy as np


import torch
from time import time as timer
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torchgeo.datasets.utils import download_and_extract_archive
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples, utils
from torchgeo.transforms import indices
from torchgeo.transforms import AugmentationSequential


import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
import torchvision.transforms.functional as F
from skimage.measure import block_reduce
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential
from tqdm import tqdm

from arguments import eval_parser
from losses import get_loss
from data import CleanFillValues, ApplyScalingAndOffset, MinMaxScaleLST, MinMaxNormalizeChannel, FillNaN, CustomRasterDataset
from utils import new_log, to_cuda, seed_all
from model import GADBase_LST


# Combine transformations
train_image_transforms = AugmentationSequential(
    CleanFillValues(fill_value=-32768),
    ApplyScalingAndOffset(scale_factor=0.01, offset=273.1499938964844),
    MinMaxScaleLST(min_val=217.65, max_val=338.18),
    data_keys=["image"]  # Ensure the transforms apply only to the 'image' key
)


# Min and max values per channel
mins = [10.0, 0.0, -71.5]
maxs = [220.0, 66.0, 5263]

# Per-channel normalization — split and apply separately
dem_transforms = AugmentationSequential(
    FillNaN(fill_value=0.0),
    MinMaxNormalizeChannel(mins[2], maxs[2]),
    data_keys=["image"] # Channel 3
)

veg_transforms = AugmentationSequential(
    FillNaN(fill_value=0.0),
    MinMaxNormalizeChannel(mins[1], maxs[1]),
    data_keys=["image"] # Channel 2
)

lc_transform = AugmentationSequential(
    FillNaN(fill_value=0.0),
    MinMaxNormalizeChannel(mins[0], maxs[0]),
    data_keys=["image"] # Channel 1
)

dem = RasterDataset(paths=(Path('Dataset/dem_aligned/')).as_posix(),
    crs='epsg:4326', res=0.01, transforms=dem_transforms
)
#dem_sampler = RandomGeoSampler(dem, size=512, length=13)

veg = RasterDataset(paths=(Path('Dataset/vegetation_aligned/')).as_posix(),
    crs='epsg:4326', res=0.01, transforms=veg_transforms
)

lc = RasterDataset(paths=(Path('Dataset/land_cover_aligned/')).as_posix(),
    crs='epsg:4326', res=0.01, transforms = lc_transform
)

guide = lc & veg & dem

eval_images = CustomRasterDataset(paths=(Path('/storage/homefs/sd22f759/SR_Pan_Arctic/prep_esa_cci_data/Dataset/lst_eval/True')).as_posix(),
     crs='epsg:4326', res=0.01, transforms=train_image_transforms, scaling=5
)

# evaluation dataset
eval_dst = eval_images & guide
eval_sampler = GridGeoSampler(eval_dst, size=240, stride=180)

# Adjust the batch size according to your GPU memory
eval_dataloader = DataLoader(
    eval_dst, sampler=eval_sampler, batch_size=8, collate_fn=stack_samples
)




class Evaluator:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        print('start with dataloader')

        self.dataloader = self.get_dataloaders(args)
        print('got data')
        
        self.model = GADBase_LST(args.feature_extractor, Npre=args.Npre, Ntrain=args.Ntrain)
        self.resume(path=args.checkpoint)
        self.model.cuda().eval()


    def evaluate(self):
        test_stats = defaultdict(float)
        num_samples = 0 

        self.model.eval()

        with torch.no_grad():
            # i is the batch number
           for i, sample in enumerate(tqdm(self.dataloader, leave=False)):
                sample = to_cuda(sample)
                #print(sample)
                #print(sample['lst'].shape)
                #print(sample['bbox'])
                #print(sample['bounds'])
                
                output = self.model(sample)
                _, loss_dict = get_loss(output, sample)
                #print(output)

                for key in loss_dict:
                    test_stats[key] += loss_dict[key]
                    
                # Save the patches as GeoTIFF after evaluation
                self.save_evaluated_patches_as_geotiffs(sample, output, i)

        return {k: v / len(self.dataloader) for k, v in test_stats.items()}

    def save_patch_as_geotiff(self, patch, output_dir, flag, batch_number, image_index, patch_index):
        bbox = patch['bbox']
        guide = patch['y_pred']
        crs = patch['crs']

        print(f"Processing patch {image_index} from image {patch_index} (batch number: {batch_number})")
        print(f"Bounding box: {bbox}")
        print(f"Guide shape: {guide.shape}")

        # Ensure the guide data is in the correct shape (bands, height, width)
        if isinstance(guide, torch.Tensor):
            guide = guide.cpu().numpy()

        # If guide is 2D, add a new axis to make it (1, height, width)
        if guide.ndim == 2:
            guide = guide[np.newaxis, ...]  # Add a channel dimension, shape becomes (1, height, width)
        elif guide.ndim == 3:
            # Already 3D (bands, height, width), no modification needed
            pass
        else:
            raise ValueError(f"Unexpected guide shape: {guide.shape}")

        # Define the scaling parameters
        min_val=217.65
        max_val=338.18

        # Reverse the scaling
        guide = guide * (max_val - min_val) + min_val

        #print(guide)

        # Ensure guide is in float32 format
        guide = guide.astype(np.float32)
        

        # Flip the image vertically
        # guide = np.flip(guide, axis=1)
        # print(guide)

        # bbox_size = (height, width)
        # bbox = [left, bottom, right, top]
        # transform = rasterio.transform.from_bounds(*bbox, width=bbox_size[1], height=bbox_size[0])

        # Extract bounding box coordinates
        minx, miny, maxx, maxy = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        print(guide.shape)

        transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, guide.shape[2], guide.shape[1])

        # Filename using both global_patch_index and image_index
        output_filename = os.path.join(output_dir, f'{flag}_{np.round(minx)}_{np.round(miny)}_classic-sound-true.tif')

        # Write the GeoTIFF
        with rasterio.open(
            output_filename,
            'w',
            driver='GTiff',
            height=guide.shape[2],
            width=guide.shape[1],
            count=1,
            dtype=guide.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(guide)

        print(f"Saved patch to {output_filename}")

    def save_evaluated_patches_as_geotiffs(self, sample, output, batch_number):
        # Set the output directory for saving patches
        output_directory = self.args.output_dir or 'output_eval_patches/'
        os.makedirs(output_directory, exist_ok=True)
        print('batch_number', batch_number)
        #print(sample)

        # Loop over each image in the sample and save patches
        #global_patch_index = 0  # Start from 0 and increment for each patch across images
        
    
        for image_index, (y_pred, bbox) in enumerate(zip(output['y_pred'], sample['bbox'][0])):
            patch = {
                    'bbox': bbox,  # Correct for each patch's bbox
                    'y_pred': y_pred,  # Correct for each patch's prediction
                    'crs': 'EPSG:4326'
                }
            # Save patch using global index and image-specific index
            global_patch_index = np.floor((batch_number + 1) / 25)
            self.save_patch_as_geotiff(patch, output_directory, 'image', batch_number, image_index, global_patch_index)
            

        # If the sample contains any bicubic outputs
        
        for image_index, (source, bbox) in enumerate(zip(sample['y_bicubic'], sample['bbox'][0])):
            for patch_index in range(len(source)):  # Iterate through the bicubic patches
                patch = {
                        'bbox': bbox,  # Correct for each patch's bbox
                        'y_pred': source,  # Correct for each bicubic patch's prediction
                        'crs': 'EPSG:4326'
                    }
                    
                # Save bicubic patch using global index and image-specific index
                global_patch_index = np.floor((batch_number + 1)*(image_index+1) / 25)
                print(batch_number,image_index, global_patch_index)
                self.save_patch_as_geotiff(patch, output_directory, 'y_bicubic', batch_number, image_index, global_patch_index)
        #except:
            #print('no bicubic')

    @staticmethod
    def get_dataloaders(args):


        eval_sampler = GridGeoSampler(eval_dst, size=args.size, stride=args.stride)
        
        eval_dataloader = DataLoader(
        eval_dst, sampler=eval_sampler, batch_size=args.batch_size, collate_fn=stack_samples
        )

        return eval_dataloader


    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')
        checkpoint = torch.load(path)
        #print(checkpoint)
        if 'model' in checkpoint:
            model_dict = checkpoint['model']
            model_dict.pop('logk2', None) # in case of using the old codebase, pop unneccesary keys
            model_dict.pop('mean_guide', None)
            model_dict.pop('std_guide', None)
            self.model.load_state_dict(model_dict)
        else:
            self.model.load_state_dict(checkpoint)
        print(f'Checkpoint \'{path}\' loaded.')

  

if __name__ == '__main__':

    args = eval_parser.parse_args()
    print(eval_parser.format_values())

    evaluator = Evaluator(args)

    since = time.time()
    stats = evaluator.evaluate()
    time_elapsed = time.time() - since

    # de-standardize losses and convert to cm (cm^2, respectively)
    #print(evaluator.dataloader.dataset.depth_transform)
    # stats, normalized temperature
    min_val=217.65
    max_val=338.18
    scale_factor = 0.01
    offset = 273.15

    # Undo min-max scaling
    stats['l1_loss'] = stats['l1_loss'] * (max_val - min_val) 
    stats['mse_loss'] = stats['mse_loss'] * (max_val - min_val) 

    print('Evaluation completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(stats)