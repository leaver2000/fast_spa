import numpy as np
from numpy.typing import ArrayLike, NDArray

def fast_spa(
    datetime_like: ArrayLike,
    elevation: ArrayLike,
    latitude: ArrayLike,
    longitude: ArrayLike,
    *,
    apply_correction: bool = False,
) -> NDArray[np.float64]: ...
def radius_vector(
    datetime_like: ArrayLike,
    *,
    apply_correction: bool = False
) -> NDArray[np.float64]: ...
def pedt(datetime_like:ArrayLike, *, apply_correction=False) -> NDArray[np.float64]: ...