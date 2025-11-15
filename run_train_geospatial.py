import os
import argparse
from collections import defaultdict
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
from torch import is_tensor, optim
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples, utils
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler
from torchgeo.transforms import AugmentationSequential
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential
from kornia.constants import DataKey
from tqdm import tqdm

from data import CleanFillValues, ApplyScalingAndOffset, MinMaxScaleLST, MinMaxNormalizeChannel, FillNaN, CustomRasterDataset
from utils import new_log, to_cuda, seed_all
from model import GADBase_LST
from losses import get_loss
from arguments import train_parser

# Combine transformations
train_image_transforms = AugmentationSequential(
    CleanFillValues(fill_value=-32768),
    ApplyScalingAndOffset(scale_factor=0.01, offset=273.1499938964844),
    MinMaxScaleLST(min_val=217.65, max_val=338.18),
    data_keys=["image"]  # Ensure the transforms apply only to the 'image' key
)

# GUIDE
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

veg = RasterDataset(paths=(Path('Dataset/vegetation_aligned/')).as_posix(),
    crs='epsg:4326', res=0.01, transforms=veg_transforms
)

lc = RasterDataset(paths=(Path('/Dataset/land_cover_aligned/')).as_posix(),
    crs='epsg:4326', res=0.01, transforms = lc_transform
)

guide = lc & veg & dem

train_images = CustomRasterDataset(paths=(Path('/Dataset/lst/')).as_posix(),
     crs='epsg:4326', res=0.01, transforms=train_image_transforms, scaling=5, split='train'
)

intersection = train_images & guide


valid_images = CustomRasterDataset(paths=(Path('Dataset/lst_val/')).as_posix(),
     crs='epsg:4326', res=0.01, transforms=train_image_transforms, scaling=5, split='test'
)

valid_dst = valid_images & guide

transform = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    data_keys=[
        DataKey.INPUT,     # 'image' 
        DataKey.INPUT,     # 'lst' 
        DataKey.INPUT,     # 'source' 
        DataKey.INPUT,      # 'mask_hr' 
        DataKey.INPUT,      # 'mask_lr' 
        DataKey.INPUT      # 'y_bicubic'
    ]
)
# Define RandomPlanckianJitter specifically for the 'image' key
guide_transform = K.AugmentationSequential(
    K.RandomPlanckianJitter(mode='blackbody', p=0.5),
    data_keys=[DataKey.INPUT]  # Only for 'image'
)

def collate_and_transform(samples):
    batch = stack_samples(samples)
    inputs = [
        batch['image'], batch['lst'], batch['source'], 
        batch['mask_hr'], batch['mask_lr'], batch['y_bicubic']
    ]
    transformed = transform(*inputs)

    (batch['image'], batch['lst'], batch['source'], 
     batch['mask_hr'], batch['mask_lr'], batch['y_bicubic']) = transformed

    batch['image'] = guide_transform(batch['image'])

    return batch


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.use_wandb = self.args.wandb
        print('start with dataloader')

        self.dataloaders = self.get_dataloaders(args)
        print('got data')
        
        seed_all(args.seed)

        self.model = GADBase_LST( 
            args.feature_extractor, 
            Npre=args.Npre,
            Ntrain=args.Ntrain, 
        ).cuda()

        self.experiment_folder, self.args.expN, self.args.randN = new_log(os.path.join(args.save_dir, 'custom'), args)
        self.args.experiment_folder = self.experiment_folder

        if self.use_wandb:
            wandb.init(project=args.wandb_project, dir=self.experiment_folder)
            wandb.config.update(self.args)
            self.writer = None
        # else:
            # self.writer = SummaryWriter(log_dir=self.experiment_folder)

        if not args.no_opt:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        else:
            self.optimizer = None
            self.scheduler = None

        self.epoch = 0
        self.iter = 0
        self.train_stats = defaultdict(lambda: np.nan)
        self.val_stats = defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf

        if args.resume is not None:
            self.resume(path=args.resume)

    def __del__(self):
        if not self.use_wandb:
            self.writer.close()

    def train(self):
        with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:
                self.train_epoch(tnr)

                if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
                    self.validate()

                    if self.args.save_model in ['last', 'both']:
                        self.save_model('last')

                if self.args.lr_scheduler == 'step':
                    if not args.no_opt:
                        self.scheduler.step()
                        if self.use_wandb:
                            wandb.log({'log_lr': np.log10(self.scheduler.get_last_lr())}, self.iter)
                        else:
                            self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()), self.epoch)

                self.epoch += 1

    def train_epoch(self, tnr=None):
        self.train_stats = defaultdict(float)

        self.model.train()
        
        # handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        # info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        # self.train_stats["gpu_used"] = info.used


        with tqdm(self.dataloaders['train'], leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)
            for i, sample in enumerate(inner_tnr):
                sample = to_cuda(sample)

                if not args.no_opt:
                    self.optimizer.zero_grad()

                output = self.model(sample, train=True)

                loss, loss_dict = get_loss(output, sample)

                if torch.isnan(loss):
                    raise Exception("detected NaN loss..")
                    
                for key in loss_dict:
                    self.train_stats[key] += loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 

                if self.epoch > 0 or not self.args.skip_first:
                    if not args.no_opt:
                        loss.backward()

                    if self.args.gradient_clip > 0.:
                        clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

                    if not args.no_opt:
                        self.optimizer.step()

                self.iter += 1

                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders['train'])) == 0:
                    self.train_stats = {k: v / self.args.logstep_train for k, v in self.train_stats.items()}

                    inner_tnr.set_postfix(training_loss=self.train_stats['optimization_loss'])
                    if tnr is not None:
                        tnr.set_postfix(training_loss=self.train_stats['optimization_loss'],
                                        validation_loss=self.val_stats['optimization_loss'],
                                        best_validation_loss=self.best_optimization_loss)

                    if self.use_wandb:
                        wandb.log({k + '/train': v for k, v in self.train_stats.items()}, self.iter)
                    else:
                        for key in self.train_stats:
                            self.writer.add_scalar('train/' + key, self.train_stats[key], self.iter)

                    # reset metrics
                    self.train_stats = defaultdict(float)

    def validate(self):
        self.val_stats = defaultdict(float)

        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(self.dataloaders['val'], leave=False):
                sample = to_cuda(sample)

                output = self.model(sample)

                loss, loss_dict = get_loss(output, sample)

                for key in loss_dict:
                    self.val_stats[key] +=  loss_dict[key].detach().cpu().item() if torch.is_tensor(loss_dict[key]) else loss_dict[key] 

            self.val_stats = {k: v / len(self.dataloaders['val']) for k, v in self.val_stats.items()}

            if self.use_wandb:
                wandb.log({k + '/val': v for k, v in self.val_stats.items()}, self.iter)
            else:
                for key in self.val_stats:
                    self.writer.add_scalar('val/' + key, self.val_stats[key], self.epoch)

            if self.val_stats['optimization_loss'] < self.best_optimization_loss:
                self.best_optimization_loss = self.val_stats['optimization_loss']
                if self.args.save_model in ['best', 'both']:
                    self.save_model('best')


    @staticmethod
    def get_dataloaders(args):

        train_sampler = RandomGeoSampler(intersection, size=args.size, length=args.length)
        valid_sampler = GridGeoSampler(valid_dst, size=args.size, stride=args.size)
        
        train_dataloader = DataLoader(
        intersection, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_and_transform, num_workers=args.num_workers)
        
        valid_dataloader = DataLoader(
        valid_dst, sampler=valid_sampler, batch_size=args.batch_size, collate_fn=stack_samples, num_workers=args.num_workers
        )

        return {'train': train_dataloader, 'val': valid_dataloader}

    def save_model(self, prefix=''):
        if args.no_opt:
            torch.save({
                'model': self.model.state_dict(),
                'epoch': self.epoch + 1,
                'iter': self.iter
            }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))
        else:
            torch.save({
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch + 1,
                'iter': self.iter
            }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if not args.no_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.iter = checkpoint['iter']

        print(f'Checkpoint \'{path}\' loaded.')

if __name__ == '__main__':
    args = train_parser.parse_args()
    print(train_parser.format_values())

    if args.wandb:
        import wandb

    trainer = Trainer(args)

    since = time.time()
    trainer.train()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
