# fast_spa

This is a Cython implementation of the NREL Solar Position Algorithm for Solar
Radiation Applications. Designed for calculating solar position across
a temporal and spatial dimension.

```python
import numpy as np
import fast_spa
import slowspa

# 200x200km area
lats = np.linspace(30, 31, 100)
lons = np.linspace(-80, -79, 100)
lats, lons = np.meshgrid(lats, lons)
datetime_obj = (
    np.arange("2023-01-01", "2023-01-02", dtype="datetime64[h]")
    .astype("datetime64[ns]")
    .astype(str)
    .tolist()
)

%timeit fast_spa.fast_spa(datetime_obj, lats, lons)
29.1 ms ± 299 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
%timeit slowspa.slow_spa(datetime_obj, lats, lons)
65 ms ± 498 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

TODO: Add a .cache to retrieve DEM cache files
