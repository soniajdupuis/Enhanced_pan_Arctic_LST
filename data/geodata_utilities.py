import abc
import functools
import glob
import os
import re
import xarray as xr
import rioxarray
from rioxarray.exceptions import NoDataInBounds
import sys
import warnings
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Optional, Union, cast
import fiona
import fiona.transform
import numpy as np
import pyproj
import rasterio
import rasterio.merge
import shapely
import torch
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rtree.index import Index, Property
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader as pil_loader
from torchgeo.datasets.utils import (BoundingBox,concat_samples, disambiguate_timestamp, merge_samples, path_is_vsi)
from torch.utils.data import DataLoader



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