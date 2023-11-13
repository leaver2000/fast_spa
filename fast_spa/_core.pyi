import datetime
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

def fast_spa(
    datetime_like: ArrayLike | Sequence[datetime.datetime],
    latitude: ArrayLike,
    longitude: ArrayLike,
    elevation: ArrayLike = ...,
    pressure: ArrayLike = ...,
    temperature: ArrayLike = ...,
    refraction: ArrayLike = ...,
    slope: ArrayLike = ...,
    azimuth_rotation: ArrayLike = ...,
    delta_t: ArrayLike | None = None,
    apply_correction: bool = False,
    num_threads: int = 1,
) -> NDArray[np.float64]: ...
def julian_ephemeris_millennium(datetime_like: ArrayLike, apply_correction: bool = False) -> NDArray[np.float64]: ...
def radius_vector(datetime_like: ArrayLike, apply_correction: bool = False) -> NDArray[np.float64]: ...
def pe4dt(datetime_like: ArrayLike, apply_correction: bool = False) -> NDArray[np.float64]: ...
def deg2xyz(degs: ArrayLike, Re: float = ...) -> NDArray[np.float64]: ...
def rad2xyz(rads: ArrayLike, Re: float = ...) -> NDArray[np.float64]: ...
def xyz2rad(xyz: ArrayLike, Re: float = ...) -> NDArray[np.float64]: ...
def xyz2deg(xyz: ArrayLike, Re: float = ...) -> NDArray[np.float64]: ...
