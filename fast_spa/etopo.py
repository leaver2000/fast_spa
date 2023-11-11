import os
import functools
from typing import Iterator, Literal, TypeVar

import numpy as np
import numpy.ma as ma
from numpy.typing import NDArray

try:
    import requests
except ImportError:
    requests = None
try:
    import netCDF4
    import pykdtree.kdtree
except ImportError:
    netCDF4 = None
    pykdtree = None
try:
    from tqdm import tqdm  # type: ignore

except ImportError:
    import contextlib

    @contextlib.contextmanager
    def tqdm(*args, **kwargs):
        s = set()
        try:
            yield s
        except:
            pass


DType_T = TypeVar("DType_T", bound=np.generic)
Methods = Literal["linear", "nearest", "cubic", "kd"]
IDX = np.ndarray | slice | int

_ETOPO_URL = "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
_ETOPO_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".fast_spa")
_ETOPO_NC_FILE = os.path.join(_ETOPO_CACHE_DIR, "ETOPO_2022_v1_60s_N90W180_surface.nc")

R = 6370997.0


def _download_etopo():
    if requests is None:
        raise ImportError("Install requests to download ETOPO2022")
    if not os.path.exists(_ETOPO_CACHE_DIR):
        os.makedirs(_ETOPO_CACHE_DIR)

    r = requests.get(_ETOPO_URL, stream=True)
    content_length = int(r.headers.get("content-length", 0))

    with open(_ETOPO_NC_FILE, "wb") as f:
        with tqdm(
            desc="Downloading ETOPO2022",
            total=content_length,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                bar.update(len(chunk))  # type: ignore
                f.write(chunk)


def _ravel(*args: NDArray[DType_T]) -> Iterator[NDArray[DType_T]]:
    return (arr.ravel() for arr in args)


def _points(lons: NDArray[np.floating], lats: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    >>> lons = np.linspace(-80, -78, 200)
    >>> lats = np.linspace(30, 31, 100)
    >>> lons, lats = np.meshgrid(lons, lats)
    >>> coords = xyz(lons.ravel(), lats.ravel())
    >>> coords.shape
    (20000, 3)
    """
    xx, yy = (np.radians(x) for x in _ravel(lons, lats))
    return np.c_[
        R * np.cos(yy) * np.cos(xx),
        R * np.cos(yy) * np.sin(xx),
        R * np.sin(yy),
    ]


class ETOPO2022:
    lat: ma.MaskedArray
    lon: ma.MaskedArray

    _file = os.path.join(_ETOPO_CACHE_DIR, "ETOPO_2022_v1_60s_N90W180_surface.nc")
    _url = "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"

    def __init__(self, file: str | None = None):
        if netCDF4 is None or pykdtree is None:
            raise ImportError("Install netCDF4 and pykdtree to use ETOPO2022")

        if not os.path.exists(_ETOPO_NC_FILE) and file is None:
            _download_etopo()

        if file is None:
            file = _ETOPO_NC_FILE

        self._ds = netCDF4.Dataset(file)  # type: ignore
        self.lon = self._ds["lon"][...]
        self.lat = self._ds["lat"][...]
        self.z = self._ds["z"]

        self._tree = functools.partial(pykdtree.kdtree.KDTree)

    def __getitem__(self, x: tuple[IDX, IDX]) -> ma.MaskedArray:
        return self.z[x]

    def resample(
        self,
        lons: NDArray,
        lats: NDArray,
        leafsize=16,
        k=3,
        eps=0,
        distance_upper_bound=None,
        sqr_dists=False,
        mask=None,
        method: Methods = "kd",
    ) -> NDArray[np.float64]:
        if lons.shape != lats.shape:
            lons, lats = np.meshgrid(lons, lats)
        assert (lons.ndim == lats.ndim == 2) & (lons.shape == lats.shape)
        y_mask = np.logical_and(self.lat >= lats.min(), self.lat <= lats.max())
        x_mask = np.logical_and(self.lon >= lons.min(), self.lon <= lons.max())
        xx, yy = np.meshgrid(self.lon[x_mask], self.lat[y_mask])
        data = self[y_mask, x_mask].ravel()
        if method != "kd":
            import scipy.interpolate

            points = _ravel(xx, yy)
            return scipy.interpolate.griddata(points, data, (lons, lats), method=method)

        shape = lons.shape
        _, i = self._tree(_points(xx, yy), leafsize=leafsize).query(
            _points(lons, lats),
            k=k,
            eps=eps,
            distance_upper_bound=distance_upper_bound,
            sqr_dists=sqr_dists,
            mask=mask,
        )
        return np.mean(data[i], axis=1).reshape(shape)
