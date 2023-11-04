import json
import numpy as np
import _spa
import spa
import pandas as pd
import pvlib
import timeit


def main():
    #

    with open("ddata.json", "r") as f:
        data = json.load(f)

    lats = np.array(data["lats"])
    lons = np.array(data["lons"])
    timestamp = pd.DatetimeIndex(data["time"])
    unixtime = (timestamp.to_numpy().astype(np.int64) // 10**9).astype(np.float64)
    print(lons[0, :].shape, lats[:, 0].shape)
    args = (
        unixtime[:1],
        lats,
        lons,
        0.0,
        # np.array([[0.0]]),
        101325.0 / 100.0,
        12.0,
        0.5667,
        spa.calculate_deltat(timestamp[0].year, timestamp[0].month),
    )

    pv_spa = np.stack(pvlib.spa.solar_position_numpy(*args, numthreads=1)[:-1])
    fast_spa = _spa.fast_spa(*args)
    print(pv_spa.shape, fast_spa.shape)
    assert np.allclose(fast_spa[:, 0], pv_spa)

    print(
        # fast_spa
        pv_spa.shape,
        fast_spa.shape,
        timeit.timeit(lambda: _spa.fast_spa(*args), number=2),
        timeit.timeit(lambda: pvlib.spa.solar_position_numpy(*args, numthreads=1), number=2),
        _spa.fast_spa(
            unixtime[:2],
            lats,
            lons,
            0.0,
            # np.array([[0.0]]),
            101325.0 / 100.0,
            12.0,
            0.5667,
            spa.calculate_deltat(timestamp[0].year, timestamp[0].month),
        ).shape,
    )


if __name__ == "__main__":
    main()
