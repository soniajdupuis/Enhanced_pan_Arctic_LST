import abc
import functools
import glob
import os
import re
import sys
import warnings
from typing import Any, Callable, Optional, Union, cast
from collections.abc import Iterable, Sequence

import numpy as np
import pyproj
import rasterio
import rasterio.merge
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
import shapely
import fiona
import fiona.transform
import rioxarray
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential
from torchvision.transforms import Normalize
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchgeo.samplers import GridGeoSampler, Units, get_random_bounding_box
from torchgeo.datasets import RasterDataset, stack_samples, unbind_samples
from torchgeo.datasets.utils import (BoundingBox, concat_samples, disambiguate_timestamp, merge_samples, path_is_vsi)
from skimage.measure import block_reduce
from rtree.index import Index, Property
from rioxarray.exceptions import NoDataInBounds
import xarray as xr
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader as pil_loader

# Additional environment configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Imports from your module
from .utils import downsample, bicubic_with_mask

class GeoDataset_2(Dataset[dict[str, Any]], abc.ABC):
    """
    source : https://github.com/noahgolmant/torchgeo/blob/c8eb26b1ae317dba9627e6f58a5ba607030f9c0a/torchgeo/datasets/geo.py
     https://github.com/microsoft/torchgeo/compare/main...noahgolmant:torchgeo:noah/xarray?expand=1

    Abstract base class for datasets containing geospatial information.
    Geospatial information includes things like:
    * coordinates (latitude, longitude)
    * :term:`coordinate reference system (CRS)`
    * resolution
    :class:`GeoDataset` is a special class of datasets. Unlike :class:`NonGeoDataset`,
    the presence of geospatial information allows two or more datasets to be combined
    based on latitude/longitude. This allows users to do things like:
    * Combine image and target labels and sample from both simultaneously
      (e.g., Landsat and CDL)
    * Combine datasets for multiple image sources for multimodal learning or data fusion
      (e.g., Landsat and Sentinel)
    * Combine image and other raster data (e.g., elevation, temperature, pressure)
      and sample from both simultaneously (e.g., Landsat and Aster Global DEM)
    These combinations require that all queries are present in *both* datasets,
    and can be combined using an :class:`IntersectionDataset`:
    .. code-block:: python
       dataset = landsat & cdl
    Users may also want to:
    * Combine datasets for multiple image sources and treat them as equivalent
      (e.g., Landsat 7 and Landsat 8)
    * Combine datasets for disparate geospatial locations
      (e.g., Chesapeake NY and PA)
    These combinations require that all queries are present in *at least one* dataset,
    and can be combined using a :class:`UnionDataset`:
    .. code-block:: python
       dataset = landsat7 | landsat8
    """

    paths: Union[str, Iterable[str]]
    paths: Optional[Union[str, Iterable[str]]] = None
    _crs = CRS.from_epsg(4326)
    _res = 0.0

    #: Glob expression used to search for files.
    #:
    #: This expression should be specific enough that it will not pick up files from
    #: other datasets. It should not include a file extension, as the dataset may be in
    #: a different file format than what it was originally downloaded as.
    filename_glob = "*"
    # NOTE: according to the Python docs:
    #
    # * https://docs.python.org/3/library/exceptions.html#NotImplementedError
    #
    # the correct way to handle __add__ not being supported is to set it to None,
    # not to return NotImplemented or raise NotImplementedError. The downside of
    # this is that we have no way to explain to a user why they get an error and
    # what they should do instead (use __and__ or __or__).
    #: :class:`GeoDataset` addition can be ambiguous and is no longer supported.
    #: Users should instead use the intersection or union operator.
    __add__ = None  # type: ignore[assignment]
    def __init__(
        self, transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None
    ) -> None:
        """Initialize a new GeoDataset instance.
        Args:
            transforms: a function/transform that takes an input sample
                and returns a transformed version
        """
        self.transforms = transforms
        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))
    @abc.abstractmethod
    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.
        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
        Returns:
            sample of image/mask and metadata at that index
        Raises:
            IndexError: if query is not found in the index
        """
    def __and__(self, other: "GeoDataset") -> "IntersectionDataset":
        """Take the intersection of two :class:`GeoDataset`.
        Args:
            other: another dataset
        Returns:
            a single dataset
        Raises:
            ValueError: if other is not a :class:`GeoDataset`
        .. versionadded:: 0.2
        """
        return IntersectionDataset(self, other)
    def __or__(self, other: "GeoDataset") -> "UnionDataset":
        """Take the union of two GeoDatasets.
        Args:
            other: another dataset
        Returns:
            a single dataset
        Raises:
            ValueError: if other is not a :class:`GeoDataset`
        .. versionadded:: 0.2
        """
        return UnionDataset(self, other)
    def __len__(self) -> int:
        """Return the number of files in the dataset.
        Returns:
            length of the dataset
        """
        return len(self.index)
    def __str__(self) -> str:
        """Return the informal string representation of the object.
        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: GeoDataset
    bbox: {self.bounds}
    size: {len(self)}"""
    # NOTE: This hack should be removed once the following issue is fixed:
    # https://github.com/Toblerity/rtree/issues/87
    def __getstate__(
        self,
    ) -> tuple[dict[str, Any], list[tuple[Any, Any, Optional[Any]]]]:
        """Define how instances are pickled.
        Returns:
            the state necessary to unpickle the instance
        """
        objects = self.index.intersection(self.index.bounds, objects=True)
        tuples = [(item.id, item.bounds, item.object) for item in objects]
        return self.__dict__, tuples
    def __setstate__(
        self,
        state: tuple[
            dict[Any, Any],
            list[tuple[int, tuple[float, float, float, float, float, float], str]],
        ],
    ) -> None:
        """Define how to unpickle an instance.
        Args:
            state: the state of the instance when it was pickled
        """
        attrs, tuples = state
        self.__dict__.update(attrs)
        for item in tuples:
            self.index.insert(*item)
    @property
    def bounds(self) -> BoundingBox:
        """Bounds of the index.
        Returns:
            (minx, maxx, miny, maxy, mint, maxt) of the dataset
        """
        return BoundingBox(*self.index.bounds)
    @property
    def crs(self) -> CRS:
        """:term:`coordinate reference system (CRS)` of the dataset.
        Returns:
            The :term:`coordinate reference system (CRS)`.
        """
        return self._crs
    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        """Change the :term:`coordinate reference system (CRS)` of a GeoDataset.
        If ``new_crs == self.crs``, does nothing, otherwise updates the R-tree index.
        Args:
            new_crs: New :term:`coordinate reference system (CRS)`.
        """
        if new_crs == self.crs:
            return
        print(f"Converting {self.__class__.__name__} CRS from {self.crs} to {new_crs}")
        new_index = Index(interleaved=False, properties=Property(dimension=3))
        project = pyproj.Transformer.from_crs(
            pyproj.CRS(str(self.crs)), pyproj.CRS(str(new_crs)), always_xy=True
        ).transform
        for hit in self.index.intersection(self.index.bounds, objects=True):
            old_minx, old_maxx, old_miny, old_maxy, mint, maxt = hit.bounds
            old_box = shapely.geometry.box(old_minx, old_miny, old_maxx, old_maxy)
            new_box = shapely.ops.transform(project, old_box)
            new_minx, new_miny, new_maxx, new_maxy = new_box.bounds
            new_bounds = (new_minx, new_maxx, new_miny, new_maxy, mint, maxt)
            new_index.insert(hit.id, new_bounds, hit.object)
        self._crs = new_crs
        self.index = new_index
    @property
    def res(self) -> float:
        """Resolution of the dataset in units of CRS.
        Returns:
            The resolution of the dataset.
        """
        return self._res
    @res.setter
    def res(self, new_res: float) -> None:
        """Change the resolution of a GeoDataset.
        Args:
            new_res: New resolution.
        """
        if new_res == self.res:
            return
        print(f"Converting {self.__class__.__name__} res from {self.res} to {new_res}")
        self._res = new_res
    @property
    def files(self) -> list[str]:
        """A list of all files in the dataset.
        Returns:
            All files in the dataset.
        .. versionadded:: 0.5
        """
        # Make iterable
        if isinstance(self.paths, str):
            paths: Iterable[str] = [self.paths]
        else:
            paths = self.paths
        # Using set to remove any duplicates if directories are overlapping
        files: set[str] = set()
        for path in paths:
            if os.path.isdir(path):
                pathname = os.path.join(path, "**", self.filename_glob)
                files |= set(glob.iglob(pathname, recursive=True))
            elif os.path.isfile(path) or path_is_vsi(path):
                files.add(path)
            else:
                warnings.warn(
                    f"Could not find any relevant files for provided path '{path}'. "
                    f"Path was ignored.",
                    UserWarning,
                )
        # Sort the output to enforce deterministic behavior.
        return sorted(files)


class RioxarrayDataset(GeoDataset_2):
    """Abstract base class for :class:`GeoDataset` stored as a single xarray dataset.
    https://github.com/microsoft/torchgeo/compare/main...noahgolmant:torchgeo:noah/xarray?expand=1
    https://github.com/noahgolmant/torchgeo/blob/c8eb26b1ae317dba9627e6f58a5ba607030f9c0a/torchgeo/datasets/geo.py
    """

    #: Names of all available bands in the dataset
    all_bands: list[str] = []

    #: Names of RGB bands in the dataset, used for plotting
    rgb_bands: list[str] = []

    #: Color map for the dataset, used for plotting
    cmap: dict[int, tuple[int, int, int, int]] = {}

    #: rioxarray x dimension name
    spatial_x_name: str = "x"

    #: rioxarray y dimension name
    spatial_y_name: str = "y"

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the dataset (overrides the dtype of the data file via a cast).
        Defaults to float32 if :attr:`~RioxarrayDataset.is_image` is True, else long.
        Can be overridden for tasks like pixel-wise regression where the mask should be
        float32 instead of long.
        Returns:
            the dtype of the dataset
        .. versionadded:: 5.0
        """
        if self.is_image:
            return torch.float32
        else:
            return torch.long

    def __init__(
        self,
        dset: xr.Dataset,
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        spatial_x_name: Optional[str] = None,
        spatial_y_name: Optional[str] = None,
        is_image: bool = True,  # Initialize is_image here
        
        
    ) -> None:
        """Initialize a new RasterDataset instance.
        Args:
            dset: xarray dataset to load.
            crs: :term:`coordinate reference system (CRS)` to warp to.
                Note: this reprojection will likely be slow and memory-intensive,
                since rioxarray loads the entire dataset into memory before reprojecting.
            res: resolution of the dataset in units of CRS. Defaults to the
                derived resolution of the dataset
            bands: bands to return (defaults to all bands), corresponding to
                the names of the data variables in the xarray dataset
            transforms: a function/transform that takes an input sample
                and returns a transformed version
        Raises:
            ValueError
                If the CRS or res are not specified and cannot be derived from the dataset,
                or if the bands do not confrom to
        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        
        super().__init__(transforms)

        self.is_image = is_image  # Define the attribute
        self.paths = None
        self.bands = bands or self.all_bands

        if crs is None:
            if dset.rio.crs is None:
                raise ValueError("CRS must be specified or present in the dataset")
            crs = CRS.from_string(dset.rio.crs)
        else:
            crs = cast(CRS, crs)

        if res is None:
            if dset.rio.crs is None:
                raise ValueError(
                    "Resolution must be specified or present in the dataset"
                )
            res = dset.rio.resolution[0]
        else:
            res = cast(float, res)
            
        # Set custom spatial dimension names if provided
        if spatial_x_name is not None:
            self.spatial_x_name = spatial_x_name
        if spatial_y_name is not None:
            self.spatial_y_name = spatial_y_name

        # Set spatial index and conform with rioxarray conventions.
        dset = dset.rio.set_spatial_dims(
            x_dim=self.spatial_x_name, y_dim=self.spatial_y_name
        )
        if "time" in dset.dims:
            ordered_dims = ("time", self.spatial_y_name, self.spatial_x_name)
        else:
            ordered_dims = (self.spatial_y_name, self.spatial_x_name)
        dset = dset.transpose(*ordered_dims, ...)

        if dset.rio.crs is not None and dset.rio.crs != crs:
            print('xx')
            # NOTE: this requires loading the entire dataset into memory.
            dset = dset.rio.reproject(crs, resolution=res)

        self._crs = crs
        self._res = res
        self.dset = dset

        # Insert the whole dataset into the index, since we are not dealing with files
        minx, miny, maxx, maxy = dset.rio.bounds()
        mint = dset.time.min().values.astype(float)
        maxt = dset.time.max().values.astype(float)
        coords = (minx, maxx, miny, maxy, mint, maxt)
        print(f"Lon: {minx} to {maxx}")
        print(f"Lat: {miny} to {maxy}")
        print(f"Time: {mint} to {maxt}")

        
        self.index.insert(0, coords, dset)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.
        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
            what is mint, maxt ?
        Returns:
            sample of image/mask and metadata at that index
        Raises:
            IndexError: if query is not found in the index
        """

        minx, maxx, miny, maxy = query.minx, query.maxx, query.miny, query.maxy
        try:
            # why not use 'sel' ?
            data = self.dset.rio.clip_box(minx=minx, maxx=maxx, miny=miny, maxy=maxy)
        except NoDataInBounds:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )
        #print(data)    

        data = data.to_array(dim="band").values
        data = torch.from_numpy(data)
        data = data.to(self.dtype)

        sample = {"crs": self.crs, "bbox": query}
        if self.is_image:
            sample["image"] = data
        else:
            sample["mask"] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
        

class MinMaxNormalize(nn.Module):
    """Normalize channels to the range [0, 1] using min/max values."""

    def __init__(self, mins: torch.Tensor, maxs: torch.Tensor):
        super().__init__()
        self.register_buffer('mins', mins.view(1, -1, 1, 1))
        self.register_buffer('maxs', maxs.view(1, -1, 1, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input - self.mins) / (self.maxs - self.mins + 1e-10)

# Your min and max values
mins = torch.tensor([10.0, 0.0, 0.0])  # min values for each channel
maxs = torch.tensor([220.0, 255.0, 4361.5])  # max values for each channel

train_label_transforms = nn.Sequential(
    MinMaxNormalize(mins, maxs),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)

class MinMaxScaleLST(K.IntensityAugmentationBase2D):
    """Normalize LST data to the range [0, 1] using min/max values."""

    def __init__(self, min_val: float = 227, max_val: float = 343) -> None:
        super().__init__(p=1)
        self.min_val = min_val
        self.max_val = max_val
        self.flags = {
            'min': torch.tensor(min_val).view(1, 1, 1, 1),
            'max': torch.tensor(max_val).view(1, 1, 1, 1)
        }

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, torch.Tensor],
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (input - flags['min']) / (flags['max'] - flags['min'] + 1e-10)

# Define training transformations for the "lst" data
train_image_transforms = AugmentationSequential(
    MinMaxScaleLST(min_val=227, max_val=343)
)




class CustomRioxarrayDataset_2(RioxarrayDataset):
    def __init__(
        self,
        dset1: xr.Dataset,
        dset2: xr.Dataset,
        size: Union[tuple[float, float], float],
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands1: Optional[Sequence[str]] = None,
        bands2: Optional[Sequence[str]] = None,
    
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        spatial_x_name: Optional[str] = None,
        spatial_y_name: Optional[str] = None,
        is_image: bool = True,
        scaling:int = 4,
        split:str = 'train'
        
    ) -> None:
        """Initialize a new CustomRioxarrayDataset instance with two datasets.

        Args:
            dset1: The first xarray dataset to load.
            dset2: The second xarray dataset to load.
            crs: The coordinate reference system (CRS) to warp to.
            res: The resolution of the datasets in units of CRS.
            bands1: The bands to return from the first dataset.
            bands2: The bands to return from the second dataset.
            transforms: A function/transform that takes an input sample and returns a transformed version.
            spatial_x_name: The name of the spatial x dimension.
            spatial_y_name: The name of the spatial y dimension.
            is_image: Whether the dataset is an image (True) or a mask (False).
        """
        # Initialize the parent class with the first dataset
        self.split = split
        super().__init__(
            dset=dset1,
            crs=crs,
            res=res,
            bands=bands1,
            transforms=transforms,
            spatial_x_name=spatial_x_name,
            spatial_y_name=spatial_y_name,
            is_image=is_image,
        )
        if self.split not in ('train', 'val', 'test'):
            raise ValueError(split)
        
        # Store the second dataset and its bands
        self.dset2 = dset2
        self.bands2 = bands2 or self.all_bands
        self.size = size
        self.scaling = scaling
        self.time_steps = (0, dset1.sizes['time'] - 1) #handle time
        
        #print(self.time_steps)
        #print('x')

        # Apply the same spatial dimension setup to the second dataset
        self.dset2 = self.dset2.rio.set_spatial_dims(
            x_dim=self.spatial_x_name, y_dim=self.spatial_y_name
        )
        if "time" in self.dset2.dims:
            ordered_dims = ("time", self.spatial_y_name, self.spatial_x_name)
        else:
            ordered_dims = (self.spatial_y_name, self.spatial_x_name)
        self.dset2 = self.dset2.transpose(*ordered_dims, ...)
        
        if self.dset2.rio.crs is not None and self.dset2.rio.crs != crs:
            self.dset2 = self.dset2.rio.reproject(crs, resolution=res)

        # Store the bounds as a regular attribute
        left, bottom, right, top = self.dset.rio.bounds()
        self.dataset_bounds = BoundingBox(
            minx=left,
            maxx=right,
            miny=bottom,
            maxy=top,
            mint=0,
            maxt=0
        )
        #print(self.dataset_bounds)
        self.bb_size = (self.size[0] * self.res, self.size[1] * self.res)
        #print('y')
    

        if split in ['val', 'test']:
            self.sampler = GridGeoSampler(
                dataset=self,
                size=(self.size[0], self.size[1]),  # Your patch size
                stride=(self.size[0] // 1.1, self.size[1] // 1.1),  # 50% overlap
                units=Units.PIXELS
            )
            print('bb', len(self.sampler))
            self.bounding_boxes = list(self.sampler) * self.dset.sizes['time'] # Convert to list
            print('data_len', self.dset.sizes['time'])
            #print(self.bounding_boxes)
            print('bb', len(self.bounding_boxes))
        else:
            self.sampler = None
            self.bounding_boxes = None    

    def __getitem__(self, index: int, called: int=0) -> dict[str, Any]:
        """
        Retrieve a sample from the dataset based on the given index.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            dict[str, Any]: A dictionary containing the sample data.
        """
        if called > 32:
            print('called more than 32')
            raise ValueError


        if self.split == 'train':
            # Generate a random bounding box
            time_index = index
        
            bounding_box = get_random_bounding_box(self.bounds, self.bb_size, self.res)
            #print(bounding_box)
            #print('z')
        elif self.split in ['val', 'test']:
            time_index = int((index*self.dset.sizes['time']) / len(self.bounding_boxes))
            bounding_box = self.bounding_boxes[index]
            print('val')
            print(bounding_box)

        # Retrieve bounding box coordinates
        minx, maxx, miny, maxy = bounding_box.minx, bounding_box.maxx, bounding_box.miny, bounding_box.maxy
        try:
        # Retrieve data from the first dataset
            data1 = self.dset.isel(time=time_index).sel(
                    {self.spatial_x_name: slice(minx, maxx), self.spatial_y_name: slice(miny, maxy)})
        except NoDataInBounds:
            raise IndexError(
                    f"query: {query} not found in index with bounds: {self.bounds}"
            )
        #print('z2')    

        try:
                # Retrieve data from the second dataset
                #print(self.dset2)
                #print(minx, maxx, miny, maxy)
            
            data2 = self.dset2.sel(
                    {self.spatial_x_name: slice(minx, maxx), self.spatial_y_name: slice(miny, maxy)})
                #print('t')
        except NoDataInBounds:
                #print('l')
            raise IndexError(
                    f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # Convert datasets to arrays and tensors
        # how to handle time here ? multiply guide layer ??
        #print('x')
        data1 = torch.from_numpy(data1.to_array(dim="band").values).to(torch.float16)
        data2 = torch.from_numpy(data2.to_array(dim="band").values).to(torch.float16)
        #print(data1.size())
        #print(data2.size())

        # apply that stuff only if split = 'train'
        if self.split == 'train':
            #print('ok train')

            # Apply any additional custom transformations or processing -> make sure that data 1 and data2 follow the same cropping !!!!
            if self.transforms is not None:

                # Apply transformations
                augmented = self.transforms(data1, data2)
                #print(augmented)
                
                data1 = augmented[0].squeeze(0)
                data2 = augmented[1].squeeze(0)
                #print(data1, data2)

        data1 = train_image_transforms(data1)
        data2 = train_label_transforms(data2)
    

        #print(data1.size())
        # Kornia transforms add a 4. dimension
        source = downsample(data1, self.scaling).squeeze().unsqueeze(0)
        data1 = data1.squeeze().unsqueeze(0)
        #print(data2.size())
        data2 = data2.squeeze()
        #print(data2.size())

        

        #print(source.size())

        # high resolution / low resolution
        mask_hr = (~torch.isnan(data1)).float()
        mask_lr = (~torch.isnan(source)).float()

        data1[mask_hr == 0.] = 0.
        source[mask_lr == 0.] = 0.
        data2[torch.isnan(data2)] = 0.
        print(f"data1 shape: {data1.shape}, non-zero elements: {torch.count_nonzero(data1)}")
        print(f"mask_lr shape: {mask_lr.shape}, non-zero elements: {torch.count_nonzero(mask_lr)}")
        print(f"guide shape: {data2.shape}, non-zero elements: {torch.count_nonzero(data2)}")

        #print(f"source shape: {source.shape}")
        #print(f"mask_lr shape: {mask_lr.shape}")


        # Check for NaNs after masking operations


        if self.split == 'train' and (torch.mean(mask_lr) < 0.9 or torch.mean(mask_hr) < 0.8):
            print('too many holes', index)
            #print(torch.mean(mask_lr))
            #print(torch.mean(mask_hr))
            # omit patch due to too many depth holes, try another one
            return self.__getitem__(index, called=called + 1)

        try:
            #print(self.scaling, source.squeeze().numpy().shape,mask_lr.squeeze().numpy().shape)
    
            y_bicubic = torch.from_numpy(bicubic_with_mask(
            source.squeeze().numpy(), mask_lr.squeeze().numpy(), self.scaling)).float()
            #print(y_bicubic)
            #print(y_bicubic.size())
            print(f"y_bicubic shape before reshape: {y_bicubic.shape}")
            print(f"y_bicubic size: {y_bicubic.numel()}")
            y_bicubic = y_bicubic.reshape((1, self.size[0], self.size[1]))
            #print(y_bicubic.size())
            #print('did manage', index)
            print(f"y_bicubic shape: {y_bicubic.shape}")
                
            return {'crs': self.crs, 'bbox': bounding_box, 'guide': data2, 'lst': data1, 'source': source, 'mask_hr': mask_hr, 'mask_lr': mask_lr,
                    'im_idx': index, 'y_bicubic': y_bicubic}
                 
        except Exception as e:
            print(f'Error processing index {index}: {str(e)}')
            if self.split == 'train':
                print('Trying another random sample')
                return self.__getitem__(index, called=called + 1)
            else:  # 'val' or 'test'
                print(f'Error in {self.split} sample {index}. Return None.')
                return None

    

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        #print(self.dset.sizes['time'])
            
        if self.split in ['val', 'test']:
            return len(self.bounding_boxes)
        else:
            return self.dset.sizes['time']
