# benchmark.py
import time

import numpy as np

import fast_spa
import slowspa


def main():
    times = np.arange("2019-01-01", "2020-01-01", dtype="datetime64[h]").astype("datetime64[ns]")
    # 200x200km area
    lats = np.linspace(30, 31, 100)
    lons = np.linspace(-80, -79, 100)
    # lats, lons = np.meshgrid(lats, lons)

    start = time.time()
    a = fast_spa.fast_spa(times, lats, lons)
    end = time.time()
    print(f"fastspa: {end - start}")
    start = time.time()
    b = slowspa.slow_spa(times, lats, lons)
    end = time.time()
    print(f"slowspa: {end - start}")
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=1e-1)


if __name__ == "__main__":
    main()
