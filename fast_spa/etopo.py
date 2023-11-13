import os

from typing import Literal, TypeVar

import numpy as np
import numpy.ma as ma


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


from ._core import deg2xyz

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


def deg2points(lon, lat):
    if len(lon) != len(lat):
        x = np.array(np.meshgrid(lon, lat))
    else:
        x = np.stack((lon, lat))[..., None]
    return deg2xyz(x.reshape(2, -1)).T


class ETOPO2022:
    lat: ma.MaskedArray
    lon: ma.MaskedArray

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

    def __getitem__(self, x: tuple[IDX, IDX]) -> ma.MaskedArray:
        return self.z[x]

    def resample(
        self,
        lons,
        lats,
        leafsize=16,
        k=3,
        eps=0,
        distance_upper_bound=None,
        sqr_dists=False,
        mask=None,
    ):
        x, y = self.lon, self.lat
        lons, lats = np.array(lons), np.array(lats)
        if lons.ndim != lats.ndim:
            raise ValueError("lons and lats must have the same number of dimensions")
        shape = (len(lats), len(lons)) if lons.ndim == 1 == lats.ndim else lons.shape
        x_mask = np.logical_and(x >= lons.min(), x <= lons.max())
        y_mask = np.logical_and(y >= lats.min(), y <= lats.max())
        src = deg2points(x[x_mask], y[y_mask])
        tgt = deg2points(lons, lats)

        data = self[y_mask, x_mask].ravel()

        tree = pykdtree.kdtree.KDTree(src, leafsize=leafsize)  # type: ignore
        _, i = tree.query(
            tgt,
            k=k,
            eps=eps,
            distance_upper_bound=distance_upper_bound,
            sqr_dists=sqr_dists,
            mask=mask,
        )
        return np.mean(data[i], axis=1).reshape(shape)
