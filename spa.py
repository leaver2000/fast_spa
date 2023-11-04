from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
import json


def calculate_deltat(year, month):
    """Calculate the difference between Terrestrial Dynamical Time (TD)
    and Universal Time (UT).

    Note: This function is not yet compatible for calculations using
    Numba.

    Equations taken from http://eclipse.gsfc.nasa.gov/SEcat5/deltatpoly.html
    """

    y = year + (month - 0.5) / 12

    deltat = np.where(year < -500, -20 + 32 * ((y - 1820) / 100) ** 2, 0)

    deltat = np.where(
        (-500 <= year) & (year < 500),
        10583.6
        - 1014.41 * (y / 100)
        + 33.78311 * (y / 100) ** 2
        - 5.952053 * (y / 100) ** 3
        - 0.1798452 * (y / 100) ** 4
        + 0.022174192 * (y / 100) ** 5
        + 0.0090316521 * (y / 100) ** 6,
        deltat,
    )

    deltat = np.where(
        (500 <= year) & (year < 1600),
        1574.2
        - 556.01 * ((y - 1000) / 100)
        + 71.23472 * ((y - 1000) / 100) ** 2
        + 0.319781 * ((y - 1000) / 100) ** 3
        - 0.8503463 * ((y - 1000) / 100) ** 4
        - 0.005050998 * ((y - 1000) / 100) ** 5
        + 0.0083572073 * ((y - 1000) / 100) ** 6,
        deltat,
    )

    deltat = np.where(
        (1600 <= year) & (year < 1700),
        120 - 0.9808 * (y - 1600) - 0.01532 * (y - 1600) ** 2 + (y - 1600) ** 3 / 7129,
        deltat,
    )

    deltat = np.where(
        (1700 <= year) & (year < 1800),
        8.83
        + 0.1603 * (y - 1700)
        - 0.0059285 * (y - 1700) ** 2
        + 0.00013336 * (y - 1700) ** 3
        - (y - 1700) ** 4 / 1174000,
        deltat,
    )

    deltat = np.where(
        (1800 <= year) & (year < 1860),
        13.72
        - 0.332447 * (y - 1800)
        + 0.0068612 * (y - 1800) ** 2
        + 0.0041116 * (y - 1800) ** 3
        - 0.00037436 * (y - 1800) ** 4
        + 0.0000121272 * (y - 1800) ** 5
        - 0.0000001699 * (y - 1800) ** 6
        + 0.000000000875 * (y - 1800) ** 7,
        deltat,
    )

    deltat = np.where(
        (1860 <= year) & (year < 1900),
        7.62
        + 0.5737 * (y - 1860)
        - 0.251754 * (y - 1860) ** 2
        + 0.01680668 * (y - 1860) ** 3
        - 0.0004473624 * (y - 1860) ** 4
        + (y - 1860) ** 5 / 233174,
        deltat,
    )

    deltat = np.where(
        (1900 <= year) & (year < 1920),
        -2.79
        + 1.494119 * (y - 1900)
        - 0.0598939 * (y - 1900) ** 2
        + 0.0061966 * (y - 1900) ** 3
        - 0.000197 * (y - 1900) ** 4,
        deltat,
    )

    deltat = np.where(
        (1920 <= year) & (year < 1941),
        21.20 + 0.84493 * (y - 1920) - 0.076100 * (y - 1920) ** 2 + 0.0020936 * (y - 1920) ** 3,
        deltat,
    )

    deltat = np.where(
        (1941 <= year) & (year < 1961),
        29.07 + 0.407 * (y - 1950) - (y - 1950) ** 2 / 233 + (y - 1950) ** 3 / 2547,
        deltat,
    )

    deltat = np.where(
        (1961 <= year) & (year < 1986),
        45.45 + 1.067 * (y - 1975) - (y - 1975) ** 2 / 260 - (y - 1975) ** 3 / 718,
        deltat,
    )

    deltat = np.where(
        (1986 <= year) & (year < 2005),
        63.86
        + 0.3345 * (y - 2000)
        - 0.060374 * (y - 2000) ** 2
        + 0.0017275 * (y - 2000) ** 3
        + 0.000651814 * (y - 2000) ** 4
        + 0.00002373599 * (y - 2000) ** 5,
        deltat,
    )

    deltat = np.where(
        (2005 <= year) & (year < 2050), 62.92 + 0.32217 * (y - 2000) + 0.005589 * (y - 2000) ** 2, deltat
    )

    deltat = np.where((2050 <= year) & (year < 2150), -20 + 32 * ((y - 1820) / 100) ** 2 - 0.5628 * (2150 - y), deltat)
    deltat = np.where(year >= 2150, -20 + 32 * ((y - 1820) / 100) ** 2, deltat)
    deltat = deltat.item() if np.isscalar(year) & np.isscalar(month) else deltat

    return deltat


def solar_position(
    timestamp: pd.DatetimeIndex,
    lat: NDArray[np.float_],
    lon: NDArray[np.float_],
    elev=0.0,
    pressure=101325.0 / 100.0,
    temp=12.0,
    atmos_refract=0.5667,
) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
    unixtime = to_timestamp(
        timestamp
    )  # np.array([timestamp.timestamp(), (timestamp + np.timedelta64(1, "D")).timestamp()])
    delta_t = calculate_deltat(timestamp.year, timestamp.month)
    jd = julian_day(unixtime)
    jde = julian_ephemeris_day(jd, delta_t)
    jc = julian_century(jd)
    jce = julian_ephemeris_century(jde)
    jme = julian_ephemeris_millennium(jce)
    heliocentric = HelioCentric(jme)
    geocentric = heliocentric.to_geocentric()
    Theta = geocentric.longitude
    beta = geocentric.latitude
    x0 = mean_elongation(jce)
    x1 = mean_anomaly_sun(jce)
    x2 = mean_anomaly_moon(jce)
    x3 = moon_argument_latitude(jce)
    x4 = moon_ascending_longitude(jce)
    l_o_nutation = np.empty((2, len(x0)))
    longitude_obliquity_nutation(jce, x0, x1, x2, x3, x4, l_o_nutation)
    delta_psi = l_o_nutation[0]
    delta_epsilon = l_o_nutation[1]
    epsilon0 = mean_ecliptic_obliquity(jme)
    epsilon = true_ecliptic_obliquity(epsilon0, delta_epsilon)
    delta_tau = aberration_correction(heliocentric.radius_vector)
    lamd = apparent_sun_longitude(Theta, delta_psi, delta_tau)
    v0 = mean_sidereal_time(jd, jc)
    v = apparent_sidereal_time(v0, delta_psi, epsilon)
    alpha = geocentric.sun_right_ascension(lamd, epsilon, beta)
    delta = geocentric.sun_declination(lamd, epsilon, beta)
    m = sun_mean_longitude(jme)
    # eot = equation_of_time(m, alpha, delta_psi, epsilon)
    H = local_hour_angle(v, lon, alpha)
    xi = equatorial_horizontal_parallax(heliocentric.radius_vector)
    term = Termination(lat, elev)

    delta_alpha = parallax_sun_right_ascension(term.x, xi, H, delta)
    delta_prime = topocentric_sun_declination(delta, term.x, term.y, xi, delta_alpha, H)
    H_prime = topocentric_local_hour_angle(H, delta_alpha)
    e0 = topocentric_elevation_angle_without_atmosphere(lat, delta_prime, H_prime)
    delta_e = atmospheric_refraction_correction(pressure, temp, e0, atmos_refract)

    e = topocentric_elevation_angle(e0, delta_e)
    theta = topocentric_zenith_angle(e)

    return theta, e
    # return theta, theta0, e, e0, phi
