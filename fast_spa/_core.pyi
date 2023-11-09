import numpy as np
from numpy.typing import ArrayLike, NDArray

def fast_spa(
    datetime_like: ArrayLike,
    latitude: ArrayLike,
    longitude: ArrayLike,
    elevation: ArrayLike | None = None,
    pressure: ArrayLike | None = None,
    temperature: ArrayLike | None = None,
    refraction: ArrayLike | None = None,
    *,
    apply_correction: bool = False,
) -> NDArray[np.float64]: ...
def julian_ephemeris_millennium(datetime_like: ArrayLike, *, apply_correction=False) -> NDArray[np.float64]: ...
def radius_vector(datetime_like: ArrayLike, *, apply_correction: bool = False) -> NDArray[np.float64]: ...
def pe4dt(datetime_like: ArrayLike, *, apply_correction=False) -> NDArray[np.float64]: ...
