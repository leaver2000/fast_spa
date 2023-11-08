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
    delta_t = fastspa.pe4dt(obj) # polinomial expression for delta_t
    assert isinstance(delta_t, np.ndarray)
    assert delta_t.dtype == np.float64
    assert delta_t.ndim == 1

    # ... slow spa
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
    unixtime = (
        np.asanyarray(obj, dtype="datetime64[ns]").astype(np.int64) // 1e9
    )
    jd = utils.julian_day(unixtime)
    assert np.all(jd == spa.julian_day(unixtime))

    jde = utils.julian_ephemeris_day(jd, delta_t)
    assert np.all(jde == spa.julian_ephemeris_day(jd, delta_t))

    jce = utils.julian_ephemeris_century(jde)

    # assert np.all(jme == spa.julian_ephemeris_millennium(jce))
    assert jme.dtype == np.float64
    assert jme.ndim == 1


@pytest.mark.parametrize("obj", date_objs)
def test_radius_vector(obj: list[Any]) -> None:
    R = fastspa.radius_vector(obj)

    assert np.allclose(
        R,
        # ... slow spa
        spa.heliocentric_radius_vector(
            fastspa.julian_ephemeris_millennium(obj)
        ),
    )


@pytest.mark.parametrize("obj", date_objs)
def test_fastspa(obj: list[Any]) -> None:
    # z = np.array([0.0])
    x = np.linspace(-180, 180, 20)
    y = np.linspace(-90, 90, 20)
    xx, yy = np.meshgrid(x, y)
    x = fastspa.fast_spa(obj, yy, xx)
    print(x.shape)


def spa_loop(times, lat, lon, elevation, pressure, delta_t):
    x = np.stack(
        [
            np.stack(
                spa.solar_position_numpy(
                    ut,
                    lat=lat,
                    lon=lon,
                    elev=elevation,
                    pressure=pressure,
                    temp=0,
                    delta_t=delta_t[i : i + 1],
                    atmos_refract=0,
                    numthreads=None,
                    sst=False,
                    esd=False,
                )[:-1]
            )
            for i, ut in enumerate(times)
        ],
        axis=1,
    )
    return x


# @pytest.mark.parametrize("obj", date_objs)
def test_f():
    obj = ["2022"]
    pressure = 1013.25
    elevation = np.array([0])
    lon = np.array([0])
    lat = np.array([0])

    dt = np.asanyarray(obj, dtype="datetime64[ns]")
    year = dt.astype("datetime64[Y]").astype(int) + 1970
    month = dt.astype("datetime64[M]").astype(int) % 12 + 1
    ut = dt.astype(np.float64) // 1e9
    delta_t = spa.calculate_deltat(year, month)
    x = spa_loop(ut, lat, lon, elevation, pressure, delta_t)
    # theta_, theta0_, e_, e0_, phi_ = x
    y = fastspa.fast_spa(obj, lat, lon, elevation)  # .ravel()
    # theta, theta0, e, e0, phi = y

    # assert y.ndim == 4
    # assert y.shape == (5, len(dt), len(lat), len(lon))
    print(
        f"""--------------
{np.allclose(x, y)}
spa
{y.ravel()}
fastspa
{x.ravel()}
"""
    )
    # fastspa.fast_spa(obj, )
    # jme = julian_ephemeris_millennium(ut, delta_t)

    # # 3.5. Calculate the true obliquity of the ecliptic
    # test_earth_heliocentric_longitude_latitude_and_radius_vector(jme)
    # L, B, R = _calculate_the_earth_heliocentric_longitude_latitude_and_radius_vector(jme)

    # # 3.4. Calculate the nutation in longitude and obliquity
    # # ∆ψ = (ai + bi * JCE ) *sin( ∑ X j *Yi, j )
    # # ∆ε = (ci + di * JCE ) *cos( ∑ X j *Yi, j )
    # test_calculate_the_nutation_in_longitude_and_obliquity(jme)
    # delta_psi, delta_epsilon = _calculate_the_nutation_in_longitude_and_obliquity(jme)

    # # 3.5. Calculate the true obliquity of the ecliptic
    # e = _true_obliquity_of_the_ecliptic(jme, delta_epsilon)

    # # 3.6-3.7.	 Calculate the apparent sun longitude, 8 (in degrees):
    # test_calculate_the_right_ascension_and_declination(L, B, R, e, delta_psi)
    # alpha, delta = _calculate_the_right_ascension_and_declination(L, B, R, e, delta_psi)
    # # print(alpha, delta)


# spa.solar_position_loop()
