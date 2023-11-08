# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=False

# pyright: reportGeneralTypeIssues=false
cimport cython
from cython.parallel cimport prange
from libc.math cimport (
    sin, cos, sqrt, atan2, asin, acos, fabs, fmod, floor, ceil, tan, pi
)
cimport numpy as cnp
import numpy as np

ctypedef unsigned long long u64

# =============================================================================
cdef inline double deg2rad(double deg) noexcept nogil: # type: ignore
    return deg * (pi / 180)

cdef inline double radians(double deg) noexcept nogil: # type: ignore
    return deg * (pi / 180)

@cython.cdivision(True)
cdef inline double rad2deg(double rad) noexcept nogil: # type: ignore
    return (rad * 180) / pi

@cython.cdivision(True)
cdef inline double degrees(double rad) noexcept nogil: # type: ignore
    return (rad * 180) / pi

cdef inline double arctan(double x) noexcept nogil: # type: ignore
    return atan2(x, 1)

cdef inline double arcsin(double x) noexcept nogil: # type: ignore
    return asin(x)

# =============================================================================
# - time
# =============================================================================
@cython.cdivision(True)
cdef inline double julian_day(double unixtime)noexcept nogil: # type: ignore
    return unixtime * 1.0 / 86400 + 2440587.5

@cython.cdivision(True)
cdef inline double  julian_ephemeris_day(double jd, double Dt)noexcept nogil: # type: ignore
    return jd + Dt * 1.0 / 86400

@cython.cdivision(True)
cdef inline double  julian_century(double jd)noexcept nogil: # type: ignore
    return (jd - 2451545) * 1.0 / 36525

@cython.cdivision(True)
cdef inline double  julian_ephemeris_century(double jde)noexcept nogil: # type: ignore
    return (jde - 2451545) * 1.0 / 36525

@cython.cdivision(True)
cdef inline double  julian_ephemeris_millennium(double jce)noexcept nogil: # type: ignore
    return jce * 1.0 / 10


cdef inline double apparent_sidereal_time_at_greenwich(
    double jd, double jc, double E, double Dpsi
) noexcept nogil: # type: ignore
    cdef double v0, v

    v0 = (
        280.46061837 
        + 360.98564736629 * (jd - 2451545) 
        + 0.000387933 * jc **2 
        - jc**3 / 38710000
    ) % 360

    v = v0 + Dpsi * cos(radians(E))                                             # ν = ν0 + ∆ψ * cos ε
    
    return v

cdef inline double pe4dt(
    long year, long month, bint apply_corection
) noexcept nogil: # type: ignore
    """ref https://eclipse.gsfc.nasa.gov/SEcat5/deltatpoly.html"""
    cdef double Dt, u, y, t

    y = year + (month - 0.5) / 12

    if year < -500:
        u = (y - 1820) / 100
        Dt = -20 + 32 * u**2
    elif year < 500:
        u = y / 100
        Dt = (
            10583.6
            - 1014.41 * u
            + 33.78311 * u**2
            - 5.952053 * u**3
            - 0.1798452 * u**4
            + 0.022174192 * u**5
            + 0.0090316521 * u**6
        )
    elif year < 1600:
        u = (y - 1000) / 100
        Dt = (
            1574.2
            - 556.01 * u
            + 71.23472 * u**2
            + 0.319781 * u**3
            - 0.8503463 * u**4
            - 0.005050998 * u**5
            + 0.0083572073 * u**6
        )
    elif year < 1700:
        t = y - 1600
        Dt = 120 - 0.9808 * t - 0.01532 * t**2 + t**3 / 7129
    elif year < 1800:
        t = y - 1700
        Dt = (
            8.83
            + 0.1603 * t
            - 0.0059285 * t**2
            + 0.00013336 * t**3
            - t**4 / 1174000
        )
    elif year < 1860:
        t = y - 1800
        Dt = (
            13.72
            - 0.332447 * t
            + 0.0068612 * t**2
            + 0.0041116 * t**3
            - 0.00037436 * t**4
            + 0.0000121272 * t**5
            - 0.0000001699 * t**6
            + 0.000000000875 * t**7
        )
    elif year < 1900:
        t = y - 1860
        Dt = (
            7.62
            + 0.5737 * t
            - 0.251754 * t**2
            + 0.01680668 * t**3
            - 0.0004473624 * t**4
            + t**5 / 233174
        )
    elif year < 1920:
        t = y - 1900
        Dt = (
            -2.79
            + 1.494119 * t
            - 0.0598939 * t**2
            + 0.0061966 * t**3
            - 0.000197 * t**4
        )
    elif year < 1941:
        t = y - 1920
        Dt = 21.20 + 0.84493 * t - 0.076100 * t**2 + 0.0020936 * t**3
    elif year < 1961:
        t = y - 1950
        Dt = 29.07 + 0.407 * t - t**2 / 233 + t**3 / 2547
    elif year < 1986:
        t = y - 1975
        Dt = 45.45 + 1.067 * t - t**2 / 260 - t**3 / 718
    elif year < 2005:
        t = y - 2000
        Dt = (
            63.86
            + 0.3345 * t
            - 0.060374 * t**2
            + 0.0017275 * t**3
            + 0.000651814 * t**4
            + 0.00002373599 * t**5
        )
    elif year < 2050:
        t = y - 2000
        Dt = 62.92 + 0.32217 * t + 0.005589 * t**2
    elif year < 2150:
        Dt = -20 + 32 * ((y - 1820) / 100) ** 2 - 0.5628 * (2150 - y)
    else:
        u = (y - 1820) / 100
        Dt = -20 + 32 * u**2

    if apply_corection:
        Dt -= -0.000012932 * (y - 1955) ** 2

    return Dt                                                              # ΔT


cdef inline double pres2alt(double pressure) noexcept nogil: # type: ignore
    '''
    Determine altitude from site pressure.

    Parameters
    ----------
    pressure : numeric
        Atmospheric pressure. [Pa]

    Returns
    -------
    altitude : numeric
        Altitude above sea level. [m]

    Notes
    ------
    The following assumptions are made

    ============================   ================
    Parameter                      Value
    ============================   ================
    Base pressure                  101325 Pa
    Temperature at zero altitude   288.15 K
    Gravitational acceleration     9.80665 m/s^2
    Lapse rate                     -6.5E-3 K/m
    Gas constant for air           287.053 J/(kg K)
    Relative Humidity              0%
    ============================   ================

    References
    -----------
    .. [1] "A Quick Derivation relating altitude to air pressure" from
       Portland State Aerospace Society, Version 1.03, 12/22/2004.
    '''
    cdef double alt

    alt = 44331.5 - 4946.62 * pressure ** (0.190263)

    return alt


cdef inline double alt2pres(double altitude) noexcept nogil: # type: ignore
    '''
    Determine site pressure from altitude.

    Parameters
    ----------
    altitude : numeric
        Altitude above sea level. [m]

    Returns
    -------
    pressure : numeric
        Atmospheric pressure. [Pa]

    Notes
    ------
    The following assumptions are made

    ============================   ================
    Parameter                      Value
    ============================   ================
    Base pressure                  101325 Pa
    Temperature at zero altitude   288.15 K
    Gravitational acceleration     9.80665 m/s^2
    Lapse rate                     -6.5E-3 K/m
    Gas constant for air           287.053 J/(kg K)
    Relative Humidity              0%
    ============================   ================

    References
    -----------
    .. [1] "A Quick Derivation relating altitude to air pressure" from
       Portland State Aerospace Society, Version 1.03, 12/22/2004.
    '''
    cdef double press

    press = 100 * ((44331.514 - altitude) / 11880.516) ** (1 / 0.1902632)

    return press


# =============================================================================
cdef inline double earth_periodic_term_summation(
    double a, double b, double c, double jme
) noexcept nogil: # type: ignore
    """3.2.	 Calculate the Earth heliocentric longitude, latitude, and radius 
    vector (L, B, # and R): """
    return a * cos(b + c * jme)

cdef tuple[double, double, double] longitude_latitude_and_radius_vector(        # 3.3.
    double JME
) noexcept nogil # type: ignore             
cdef tuple[double, double] nutation_in_longitude_and_obliquity(                 # 3.4.
    double JCE
) noexcept nogil # type: ignore
cdef double true_obliquity_of_the_ecliptic(         
    double JME, double delta_eps
) noexcept nogil  # type: ignore
cdef tuple[double, double] geocentric_right_ascension_and_declination(          # 3.9.
    double apparent_lon, double geocentric_lat, double true_ecliptic_obliquity
) noexcept nogil # type: ignore
# =============================================================================
cdef inline double equatorial_horizontal_parallax(                              # 3.12.1.
    double R
) noexcept nogil: # type: ignore
    cdef double xi 
    xi = 8.794 / (3600 * R)
    return xi

# =============================================================================
# 3.12. Calculate the topocentric sun right ascension "’ (in degrees): 
# 3.12.2. Calculate the term u (in radians),
cdef tuple[double,double] topocentric_declination_hour_angle(
    # double alpha,   # α geocentric sun right ascension
    double delta,   # δ geocentric sun declination
    double H,       # H local hour angle
    double E,       # E observer elevation
    double lat,     # observer latitude
    double xi,      # ξ equatorial horizontal parallax
) noexcept nogil # type: ignore