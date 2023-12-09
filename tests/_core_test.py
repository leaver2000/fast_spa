from __future__ import annotations
from typing import Any

import pytest
import numpy as np
import pvlib.spa as spa

import fast_spa._core as fastspa


date_objs = [
    ["2019-01-01 00:00:00"],
    ["2019-01-01 00:00:00", "2019-02-01 00:00:00"],
    ["2019-01-01 00:00:00", "2020-01-01 00:00:00", "2019-01-01 00:00:00"],
]


@pytest.mark.parametrize("obj", date_objs)
def test_pedt(obj: list[Any]) -> None:
    delta_t = fastspa.pe4dt(obj)  # polynomial expression for delta_t
    assert isinstance(delta_t, np.ndarray)
    assert delta_t.dtype == np.float64
    assert delta_t.ndim == 1

    # slow spa...
    dt = np.asarray(obj, dtype="datetime64[ns]")
    year = dt.astype("datetime64[Y]").astype(int) + 1970
    month = dt.astype("datetime64[M]").astype(int) % 12 + 1

    assert np.allclose(delta_t, spa.calculate_deltat(year, month))


@pytest.mark.parametrize("obj", date_objs)
def test_julian_ephemeris_millennium(obj) -> None:
    jme = fastspa.julian_ephemeris_millennium(obj)
    assert isinstance(jme, np.ndarray)
    assert jme.dtype == np.float64
    assert jme.ndim == 1

    delta_t = fastspa.pe4dt(obj)
    unix_time = np.asanyarray(obj, dtype="datetime64[ns]").astype(np.int64) // 1e9

    assert np.all(
        jme
        # slow spa...
        == spa.julian_ephemeris_millennium(
            spa.julian_ephemeris_century(spa.julian_ephemeris_day(spa.julian_day(unix_time), delta_t))
        )
    )
    assert jme.dtype == np.float64
    assert jme.ndim == 1


@pytest.mark.parametrize("obj", date_objs)
def test_radius_vector(obj: list[Any]) -> None:
    R = fastspa.radius_vector(obj)

    assert np.allclose(
        R,
        # slow spa...
        spa.heliocentric_radius_vector(fastspa.julian_ephemeris_millennium(obj)),
    )


@pytest.mark.parametrize("obj", date_objs)
def test_fastspa(obj: list[Any]) -> None:
    # z = np.array([0.0])
    x = np.linspace(-180, 180, 20)
    y = np.linspace(-90, 90, 20)
    xx, yy = np.meshgrid(x, y)
    x = fastspa.fast_spa(obj, yy, xx)
    # print(x.shape)


def solar_position_numpy(*args, **kwargs):
    theta, theta0, e, e0, phi, eot = spa.solar_position_numpy(  # type: ignore
        *args,
        **kwargs,
        sst=False,
        esd=False,
    )
    return [theta0, phi]


def slow_spa(
    obj,
    lat,
    lon,
    elevation,
    pressure,
    temp,
    refraction,
):
    delta_t = fastspa.pe4dt(obj)
    dt = np.asanyarray(obj, dtype="datetime64[ns]")
    unix_time = dt.astype(np.float64) // 1e9

    x = np.stack(
        [
            np.stack(
                solar_position_numpy(
                    ut,
                    lat=lat,
                    lon=lon,
                    elev=elevation,
                    pressure=pressure,
                    temp=temp,
                    delta_t=delta_t[i : i + 1],
                    atmos_refract=refraction,
                    numthreads=None,
                )
            )
            for i, ut in enumerate(unix_time)
        ],
        axis=1,
    )
    return x


@pytest.mark.parametrize("obj", date_objs)
def test_f(obj):
    pressure = np.array([1013.25])
    refraction = np.array([0.5667])
    temp = np.array([12])
    elevation = np.array([0])
    lon = np.array([0])
    lat = np.array([0])
    # slow spa...
    zen1, azi1 = slow_spa(obj, lat, lon, elevation, pressure, temp, refraction)
    zen, azi = fastspa.fast_spa(obj, lat, lon, elevation, pressure, temp, refraction)

    assert np.allclose(zen1, zen, atol=1e-3)
    assert np.allclose(azi1, azi, atol=1e-2)


@pytest.mark.parametrize("obj", date_objs)
def test_fast_spa_with_weird_args(obj):
    # 200x200km area
    P0 = 1013.25
    E = 0.0
    R = 0.5667
    T = 12.0

    lats = np.linspace(30, 31, 100)
    lons = np.linspace(-80, -79, 100)
    lats, lons = np.meshgrid(lats, lons)

    all_scalars = fastspa.fast_spa(
        obj,
        lats,
        lons,
        elevation=E,
        pressure=P0,
        temperature=T,
        refraction=R,
    )

    elev_array = fastspa.fast_spa(
        obj,
        lats,
        lons,
        elevation=np.full_like(lats, E),
        pressure=P0,
        temperature=T,
        refraction=R,
    )
    assert elev_array.shape == all_scalars.shape == (2, len(obj)) + lats.shape


@pytest.fixture
def spa_results():
    """
    A.5. Example
    The results for the following site parameters are listed in Table A5.1:
    - Date = October 17, 2003.
    - Time = 12:30:30 Local Standard Time (LST).
    - Time zone(TZ) = -7 hours.
    - Longitude = -105.1786/.
    - Latitude = 39.742476/.
    - Pressure = 820 mbar.
    - Elevation = 1830.14 m.
    - Temperature = 11/C.
    - Surface slope = 30/.
    - Surface azimuth rotation = -10/.
    - )T = 67 Seconds.

    LST must be changed to UT by subtracting TZ from LST, and changing the date if necessary.
    """
    return (
        ["2003-10-17 19:30:30"],
        [39.742476],  # lat
        [-105.1786],  # lon
        1830.14,  # elevation
        820.94,  # pressure
        11.0,  # temperature
        0.5667,  # refraction
        30.0,  # slope
        -10.0,  # azimuth_rotation
        67.0,  # delta_t
    ), (
        50.11162,  # zenith
        194.34024,  # azimuth
    )


def test_results_for_example(spa_results):
    args, (expect_zen, expect_azi) = spa_results
    zen, azi = fastspa.fast_spa(*args).ravel()
    assert np.allclose(zen, expect_zen, atol=1e-2)
    assert np.allclose(azi, expect_azi, atol=1e-2)


DEGS = [
    [[0, 1, 3, 4, 5], [0, 1, 3, 4, 5]],
]


@pytest.mark.parametrize("degs", DEGS)
def test_xyz_rad_deg(degs):
    degs = np.array(degs, dtype=float)
    xyz = fastspa.deg2xyz(degs)
    assert xyz.shape == (3,) + degs.shape[1:]
    rads = fastspa.xyz2rad(xyz)
    assert rads.shape == degs.shape
    assert np.allclose(degs, np.degrees(rads), atol=1e-6)
    assert np.allclose(xyz, fastspa.rad2xyz(rads), atol=1e-6)
