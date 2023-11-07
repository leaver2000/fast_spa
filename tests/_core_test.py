import numpy as np
import fastspa._core as fastspa
import fastspa._utils as utils

def test_pedt():
    t = ['2022']
    x = fastspa.pedt(t)

    dt = np.asarray(t, dtype='datetime64[ns]')
    year = dt.astype('datetime64[Y]').astype(int) + 1970
    month = dt.astype('datetime64[M]').astype(int) % 12 + 1
    y = utils.calculate_deltat(year, month)
    
    assert np.allclose(x, y)


def test_radius_vector():
    t = ['2022']
    x = fastspa.radius_vector(t)

    # utils.heliocentric_radius_vector()
    print(x.shape)


def test_fastspa():
    t = ['2022']
    z = np.array([0.0])
    x = np.linspace(-180, 180, 20)
    y = np.linspace(-90, 90, 20)
    xx, yy = np.meshgrid(x, y)
    x = fastspa.fast_spa(t, z, yy, xx)
    print(x.shape)