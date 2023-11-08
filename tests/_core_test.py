from __future__ import annotations
from typing import Any

import pytest
import numpy as np
import pvlib.spa as spa

import fastspa._core as fastspa
import fastspa._utils as utils
from fastspa._core import Julian


date_objs = [
    ["2019-01-01 00:00:00"],
    ["2019-01-01 00:00:00", "2019-02-01 00:00:00"],
    ["2019-01-01 00:00:00", "2020-01-01 00:00:00", "2019-01-01 00:00:00"],
]


@pytest.mark.parametrize("obj", date_objs)
def test_julian(obj):
    j = Julian(obj)
    dt = np.asanyarray(obj, dtype="datetime64[ns]")
    year = dt.astype("datetime64[Y]").astype(int) + 1970
    month = dt.astype("datetime64[M]").astype(int) % 12 + 1
    ut = dt.astype(np.float64) // 1e9
    assert np.all(j.day == spa.julian_day(ut))
    assert np.all(j.delta_t == spa.calculate_deltat(year, month))
    assert np.all(j.ephemeris_day == spa.julian_ephemeris_day(j.day, j.delta_t))
    assert np.all(j.ephemeris_century == spa.julian_ephemeris_century(j.ephemeris_day))
    assert np.all(j.ephemeris_millennium == spa.julian_ephemeris_millennium(j.ephemeris_century))


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
    print(x.shape)


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
                spa.solar_position_numpy(
                    ut,
                    lat=lat,
                    lon=lon,
                    elev=elevation,
                    pressure=pressure,
                    temp=temp,
                    delta_t=delta_t[i : i + 1],
                    atmos_refract=refraction,
                    numthreads=None,
                    sst=False,
                    esd=False,
                )[:-1]
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
    x = slow_spa(obj, lat, lon, elevation, pressure, temp, refraction)

    y = fastspa.fast_spa(obj, lat, lon, elevation, pressure, temp, refraction)

    # there appears to be some differences with the division
    # in the c api. specifically regarding the delta
    # δ  = arcsin(sin(B) * cos(E) + cos(B) * sin(E) * sin(A))
    assert np.allclose(x.ravel(), y.ravel(), atol=1e-2)
