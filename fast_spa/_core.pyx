# cython: language_level=3

# pyright: reportGeneralTypeIssues=false, reportUnusedExpression=false, reportMissingImports=false

# import cython
cimport cython

from cython.parallel cimport prange

import numpy as np
from numpy.typing import ArrayLike
cimport numpy as np
from numpy cimport uint16_t as u16
from numpy import (
    sin,
    cos,
    tan,
    arcsin,
    arctan,
    arctan2,
    degrees,
    radians,
)

from . cimport _lib as lib
from . cimport _terms as terms

np.import_array()
np.import_umath()
np.import_ufunc()


# =============================================================================
# - types
# =============================================================================
ctypedef np.float64_t DTYPE_t

ctypedef fused floating:
    np.float64_t
    np.float32_t


cdef fused Elevation_t:
    double
    np.ndarray[DTYPE_t, ndim=1] # type: ignore

cdef fused Pressure_t:
    double
    np.ndarray[DTYPE_t, ndim=1] # type: ignore

cdef fused Temperature_t:
    double
    np.ndarray[DTYPE_t, ndim=1] # type: ignore

cdef fused Refraction_t:
    double
    np.ndarray[DTYPE_t, ndim=1] # type: ignore

# =============================================================================
# - constants
# =============================================================================

# - time components
cdef u16 RIGHT_ASCENSION = 0
cdef u16 DECLINATION = 1
cdef u16 SIDEREAL_TIME = 2
cdef u16 HORIZONAL_PARALAX = 3


# - spatial components
cdef u16 ZENITH_ANGLE = 0
cdef u16 AZIMUTH_ANGLE = 1


cdef u16 NUM_TIME_COMPONENTS = 4
cdef u16 NUM_SPA_COMPONENTS = 2


F64:Floating
F32:Floating

cdef enum Floating:
    F32 = np.NPY_FLOAT
    F64 = np.NPY_DOUBLE
    

DTYPE = np.float64

# cdef double Re = 6370997.0
EARTH_RADIUS = 6370997.0
# =============================================================================
cdef np.ndarray asfarray(object x, Floating dtype):
    return np.PyArray_FROM_OT(x, dtype)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef unixtime_delta_t(datetime_like, bint apply_correction = 0) noexcept:
    cdef np.ndarray dt
    cdef long[:] y, m
    cdef double[:] ut, delta_t

    dt = lib.dtarray(datetime_like)
    ut = lib.unixtime(dt)
    y = lib.years(dt)
    m = lib.months(dt)
    delta_t = _pe4dt(y, m, apply_correction)

    return ut, delta_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:] _julian_ephemeris_millennium(
    double[:] unixtime, double[:] delta_t, int num_threads) noexcept nogil: # type: ignore
    cdef int n, i
    cdef double ut, dt
    cdef double[:] out

    n = len(unixtime)
    with gil:
        out = lib.view1d(n)

    for i in prange(n, nogil=True, num_threads=num_threads):
        ut  = unixtime[i]
        dt = delta_t[i]
        out[i] = (
            lib.julian_ephemeris_millennium(
                lib.julian_ephemeris_century(
                    lib.julian_ephemeris_day(lib.julian_day(ut), dt)
                    )
                )
            )

    return out


# - python interface
def julian_ephemeris_millennium(
    datetime_like, bint apply_correction = 0, int num_threads = 1):
    cdef double[:] ut, delta_t
    ut, delta_t = unixtime_delta_t(datetime_like, apply_correction)
    return np.asfarray(_julian_ephemeris_millennium(ut, delta_t, num_threads))

# =============================================================================
# Polynomila Expression for delta_t
# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:] _pe4dt(
    long[:] years, long[:] months, bint apply_corection
) noexcept nogil: # type: ignore
    cdef int n, i
    cdef long year, month
    cdef double[:] out

    n = len(years)
    with gil:
        out = lib.view1d(n)

    for i in prange(n, nogil=True):
        year = years[i]
        month = months[i]
        out[i] = lib.pe4dt(year, month, apply_corection)

    return out

# - python interface
def pe4dt(datetime_like, bint apply_correction = 0):
    cdef np.ndarray dt
    cdef long[:] y, m

    dt = lib.dtarray(datetime_like)
    y = lib.years(dt)
    m = lib.months(dt)

    return np.asfarray(_pe4dt(y, m, apply_correction))


# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:] _radius_vector(
    double[:] unixtime, double[:] delta_t, int num_threads = 1
) noexcept nogil: # type: ignore
    cdef int n, i
    cdef double[:] jme, out

    jme = _julian_ephemeris_millennium(unixtime, delta_t, num_threads)
    n = len(jme)
    with gil:
        out = lib.view1d(n)

    for i in prange(n, nogil=True, num_threads=num_threads):
        out[i] = terms.heliocentric_radius_vector(jme[i], num_threads=num_threads)

    return out

# - python interface
def radius_vector(object datetime_like, bint apply_correction=False):
    cdef double[:] ut, delta_t

    ut, delta_t = unixtime_delta_t(datetime_like, apply_correction)

    return np.asarray(_radius_vector(ut, delta_t))


# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:, :] _time_components(
    double[:] unixtime, double[:] delta_t, u16 num_threads = 1
) noexcept nogil: # type: ignore
    cdef int n, i
    cdef double ut, dt, jd, jc, jce, jme, L, B, R, O, delta_psi, delta_eps, E, delta_tau, Lambda
    cdef double[:, :] out
    n = len(unixtime)

    with gil:
        out = lib.view2d(NUM_TIME_COMPONENTS, n)

    for i in prange(n, nogil=True, num_threads=num_threads):
        ut  = unixtime[i]
        dt = delta_t[i]

        # - 3.1. Calculate the Julian and Julian Ephemeris Day, Century, and Millennium
        jd = lib.julian_day(ut)
        jc = lib.julian_century(jd)
        jce = lib.julian_ephemeris_century(lib.julian_ephemeris_day(jd, dt))
        jme = lib.julian_ephemeris_millennium(jce)

        # - 3.2 Calculate the Earth heliocentric longitude, latitude (L, B, R) 
        L = terms.heliocentric_longitude(jme, num_threads=num_threads)          # L = ∑ X j *Yi, j
        B = terms.heliocentric_latitude(jme, num_threads=num_threads)           # B = ∑ X j *Yi, j
        R = terms.heliocentric_radius_vector(jme, num_threads=num_threads)      # R = ∑ X j *Yi, j

        # - 3.3 Calculate the geocentric longitude and latitude
        O = (L + 180.0) % 360.0                                                 # Θ = L + 180 geocentric longitude (in degrees)

        # - 3.4 Calculate the nutation in longitude and obliquity
        delta_psi, delta_eps = (
            terms.nutation_in_longitude_and_obliquity(                          # ∆ψ = (ai + bi * JCE ) *sin( ∑ X j *Yi, j )
                jce, num_threads=num_threads                                    # ∆ε = (ci + di * JCE ) *cos( ∑ X j *Yi, j )    
            )
        )
        
        # - 3.5 Calculate the true obliquity of the ecliptic
        E = lib.true_obliquity_of_the_ecliptic(jme, delta_eps)                  # ε = ε0 / 3600 + ∆ε

        # - 3.6 Calculate the aberration correction (in degrees)
        delta_tau = -20.4898 / (R * 3600.0)                                     # ∆τ = − 26.4898 / 3600 * R 

        # - 3.7 Calculate the apparent sun longitude (in degrees)
        Lambda = (O + delta_psi + delta_tau) % 360.0                            # λ = Θ + ∆ψ + ∆τ

        # - 3.8. Calculate the apparent sidereal time at Greenwich (in degrees) 
        out[SIDEREAL_TIME, i] = (                                               # ν = ν0 + ∆ψ * cos ε
            lib.apparent_sidereal_time_at_greenwich(jd, jc, E, delta_psi)
        )
        
        # - 3.9,3.10 Calculate the geocentric sun right ascension & declination
        out[RIGHT_ASCENSION, i], out[DECLINATION, i] = (                        # α = ArcTan2(sin λ *cos ε − tan β *sin ε, cos λ)
            lib.geocentric_right_ascension_and_declination(Lambda, -B, E)       # δ = Arcsin(sin β *cos ε + cos β *sin ε *sin λ) 
        )

        # 3.12.1. Calculate the equatorial horizontal parallax of the sun
        # NOTE: in the name of compute this function is performed out of order
        # because it is independent of the spatial components
        out[HORIZONAL_PARALAX, i] = 8.794 / (3600 * R)                          # ξ = 8.794 / (3600 * R)
    
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double[:, :] time_components(
    object datetime_like,
    bint apply_correction = False, 
    int num_threads = 1
):
    cdef double[:] ut, delta_t
    cdef double[:, :] out

    ut, delta_t = unixtime_delta_t(datetime_like, apply_correction)
    out = _time_components(ut, delta_t, num_threads=num_threads)

    return out


# - python interface
def get_time_components(
    datetime_like: ArrayLike, apply_correction = False, int num_threads = 1):
    return np.asfarray(
        time_components(datetime_like, apply_correction, num_threads)
    )


# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef np.ndarray _fast_spa(
    double[:, :] tc,
    np.ndarray[DTYPE_t, ndim=1] lat,
    np.ndarray[DTYPE_t, ndim=1] lon,
    Elevation_t E,
    Pressure_t P,
    Temperature_t T,
    Refraction_t refct,
    # double W,                                                             # slope of the surface measured from the horizontal plane
    # double Y,                                                             # surface azimuth rotation angle
    int num_threads,
):
    cdef int i, n
    cdef double v, xi, delta, alpha                                             # time components (ν, ξ, δ, α)
    cdef np.ndarray[DTYPE_t, ndim=1] H, delta_alpha, delta_p, H_p, e0, e, gamma, phi
    cdef np.ndarray[DTYPE_t, ndim=3] out                                           # (C, T, Y*X)

    n   = tc.shape[1]
    phi = radians(lat)                                                          # ϕ
    out = np.empty((NUM_SPA_COMPONENTS, n, len(lat)), dtype=DTYPE)

    for i in prange(n, nogil=True, num_threads=num_threads):
        v       = tc[SIDEREAL_TIME, i]                                          # ν:deg
        xi      = lib.radians(tc[HORIZONAL_PARALAX, i])                         # ξ:rad
        delta   = lib.radians(tc[DECLINATION, i])                               # δ:rad
        alpha   = tc[RIGHT_ASCENSION, i]                                        # α:deg

        with gil:
            # - 3.11. Calculate the observer local hour angle
            H = (v + lon - alpha) % 360                                         # H:deg = ν + λ − α

            # - 3.12.2. Calculate the term u 
            u = arctan(0.99664719 * tan(phi))                                   # u:rad = Arctan(0.99664719 * tan ϕ)

            # - 3.12.3. Calculate the term x
            x = cos(u) + E / 6378140 * cos(phi)                                 # x:rad = cos u + E / 6378140 * cos ϕ

            # - 3.12.4. Calculate the term y
            y = 0.99664719 * sin(u) + E / 6378140 * sin(phi)                    # y:rad = 0.99664719 * sin u + E / 6378140 * sin ϕ

            # - 3.12.5. Calculate the parallax in the sun right ascension
            delta_alpha = arctan2(
                -x * sin(xi) * sin(H), cos(delta) - x * sin(xi) * cos(H)        # ∆α:rad = ArcTan2(−x * sin ξ * sin H, cos δ − x * sin ξ * cos H)
            )

            delta_p = (
                arcsin(sin(delta) - y * sin(xi) * cos(delta_alpha))             # ∆':rad = Arcsin(sinδ − y * sin ξ * cos ∆α)
            ) 

            H_p = radians(H - degrees(delta_alpha))                             # H':rad = H − ∆α

            # - 3.14.1 Calculate the topocentric elevation angle without atmospheric refraction correction
            e0 = arcsin(
                sin(phi) * sin(delta_p) + cos(phi) * cos(delta_p) * cos(H_p)    # e0:rad = Arcsin(sin ϕ *sin ∆' + cos ϕ *cos ∆' *cos H')
            ) 
            e0 = degrees(e0)                                                    # e0:deg

            # - 3.14.2 Calculate the atmospheric refraction correction
            #   - P is the annual average local pressure (in millibars).
            #   - T is the annual average local temperature (in /C). 
            #   - e0 is in degrees. Calculate the tangent argument in degrees
            #       Note that ∆e = 0 when the sun is below the horizon.
            delta_e = (                                                         # ∆e:deg = P / 1010 * 283 / 273 + T * 1.02 / (60 * tan(e0 + 10.3 / (e0 + 5.11)))
                (P / 1010) 
                * (283 / (273 + T))  
                * 1.02 / (60 * tan(radians(e0 + 10.3 / (e0 + 5.11))))
            ) * (e0 >= -1.0 * (0.26667 + refct))

            # - 3.14.3 Calculate the topocentric elevation angle
            e =  e0 + delta_e                                                   # e:deg = e0 + ∆e

            # - 3.14.4 Calculate the topocentric zenith angle
            out[ZENITH_ANGLE, i, :]  = 90 - e                                   # θ:deg = 90 − e 

            # - 3.15   Calculate the topocentric azimuth angle
            # - 3.15.1 Calculate the topocentric astronomers azimuth angle
            gamma = arctan2(                                                    # Γ:rad = ArcTan2(sin H', cos H' *sin ϕ − tan ∆' *cos ϕ)
                sin(H_p), cos(H_p) * sin(phi) - tan(delta_p) * cos(phi)
            )

            # - 3.15.2 Calculate the topocentric azimuth angle
            out[AZIMUTH_ANGLE, i, :] = (degrees(gamma) + 180) % 360     # Φ:deg = Γ + 180

    return out

# - python interface
def fast_spa(
    object datetime_like,
    object latitude,
    object longitude,
    object elevation = 0.0,
    object pressure = 1013.25,
    object temperature = 12.0,
    object refraction= 0.0,
    double slope = 0.0,
    double azimuth_rotation = 0.0,
    object delta_t = None,
    bint apply_correction = False,
    int num_threads = 1,
) -> np.ndarray:
    cdef int size
    cdef tuple shape
    cdef double[:] ut, dt
    cdef np.ndarray out, lats, lons
    cdef double[:, :] time_components

    if delta_t is None:
        ut, dt = unixtime_delta_t(datetime_like, apply_correction)
    else:
        if np.isscalar(delta_t):
            delta_t = [delta_t]

        dt = np.asfarray(delta_t)
        ut = lib.unixtime(lib.dtarray(datetime_like))

    time_components = _time_components(ut, dt, num_threads=num_threads)
    
    lats = np.asfarray(latitude, dtype=DTYPE)
    lons = np.asfarray(longitude, dtype=DTYPE)

    if not lats.ndim == lons.ndim:
        lats, lons = np.meshgrid(lats, lons)

    assert lats.ndim == lons.ndim

    shape = ()
    if lats.ndim == 2:
        shape = (NUM_SPA_COMPONENTS, time_components.shape[1], lats.shape[0], lats.shape[1])
        lats, lons = lats.ravel(), lons.ravel()
    size = lats.size

    if (
        np.isscalar(elevation) 
        and np.isscalar(pressure) 
        and np.isscalar(temperature) 
        and np.isscalar(refraction)
    ):
        out = _fast_spa[double, double, double, double](
            time_components,
            lats,
            lons,
            elevation,
            pressure,
            temperature,
            refraction,
            # slope,
            # azimuth_rotation,
            num_threads,
        )

    elif (
        not np.isscalar(elevation) 
        and np.isscalar(pressure) 
        and np.isscalar(temperature) 
        and np.isscalar(refraction)
    ):
        out = _fast_spa[np.ndarray, double, double, double](
            time_components,
            lats,
            lons,
            lib.darray1d(elevation, size),
            pressure,
            temperature,
            refraction,
            # slope,
            # azimuth_rotation,
            num_threads,
        )

    else:
        out = _fast_spa[np.ndarray, np.ndarray, np.ndarray, np.ndarray](
            time_components,
            lats,
            lons,
            lib.darray1d(elevation, size),
            lib.darray1d(pressure, size),
            lib.darray1d(temperature, size),
            lib.darray1d(refraction, size),
            # slope,
            # azimuth_rotation,
            num_threads,
        )

    if shape:
        out = out.reshape(shape)

    return out

# =============================================================================
# https://rbrundritt.wordpress.com/2008/10/14/conversion-between-spherical-and-cartesian-coordinates-systems/
cdef _rad2xyz(np.ndarray[DTYPE_t, ndim=2] rads, double Re):
    cdef np.ndarray[DTYPE_t, ndim=1] phi, theta
    cdef np.ndarray[DTYPE_t, ndim=2] xyz
    xyz = np.empty((3, rads.shape[1]), dtype=DTYPE)
    phi = rads[0]
    theta = rads[1]

    # - spherical
    xyz[0] = np.cos(theta) * np.cos(phi)
    xyz[1] = np.cos(theta) * np.sin(phi)
    xyz[2] = np.sin(theta)

    # - cartesian
    xyz *=  Re
    return xyz

cdef _xyz2rad(np.ndarray[DTYPE_t, ndim=2] xyz, double Re):
    cdef np.ndarray[DTYPE_t, ndim=1] x, y, z
    cdef np.ndarray[DTYPE_t, ndim=2] rads
    rads = np.empty((2, xyz.shape[1]), dtype=DTYPE)

    # - vertices
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    # - spherical
    rads[0] = np.arctan2(y, x)
    rads[1] = np.arcsin(z / Re)

    return rads


def rad2xyz(object rads, double Re = EARTH_RADIUS):
    a = asfarray(rads, F64)
    return _rad2xyz(a, Re)

def deg2xyz(object deg, double Re = EARTH_RADIUS):
    a = np.radians(asfarray(deg, F64))
    return _rad2xyz(a, Re)

def xyz2rad(object xyz, double Re = EARTH_RADIUS):
    a = asfarray(xyz, F64)
    return _xyz2rad(a, Re)

def xyz2deg(object xyz, double Re = EARTH_RADIUS):
    a = asfarray(xyz, F64)
    return np.degrees(_xyz2rad(a, Re))

