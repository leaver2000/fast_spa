# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=False

from cython.parallel cimport prange
import numpy as np
cimport numpy as np
from numpy.typing import NDArray, ArrayLike
import itertools
import typing

from . cimport _lib as lib, _terms as terms

np.import_array()

Boolean: typing.TypeAlias = "bool | bint"


cdef int TOPOCENTRIC_RIGHT_ASCENSION = 0
cdef int TOPOCENTRIC_DECLINATION = 1
cdef int APARENT_SIDEREAL_TIME = 2
cdef int EQUATOIRAL_HORIZONAL_PARALAX = 3
cdef int NUM_TIME_COMPONENTS = 4

    
cdef unixtime_delta_t(
    datetime_like, apply_correction: Boolean = False) noexcept:
    cdef np.ndarray dt
    cdef long[:] y, m
    cdef double[:] ut, delta_t

    dt = lib.dtarray(datetime_like)
    ut = lib.unixtime(dt)
    y = lib.years(dt)
    m = lib.months(dt)
    delta_t = _pe4dt(y, m, apply_correction)

    return ut, delta_t

    
cdef double[:] _julian_ephemeris_millennium(
    double[:] unixtime, double[:] delta_t) noexcept:
    cdef int n, i
    cdef double ut, dt
    cdef double[:] out

    n = len(unixtime)
    out = lib.view1d(n)


    for i in prange(n, nogil=True):
        ut  = unixtime[i]
        dt = delta_t[i]
        out[i] = lib.julian_ephemeris_millennium(
            lib.julian_ephemeris_century(
                lib.julian_ephemeris_day(
                    lib.julian_day(ut), dt
                    )
                )
            )
    return out


# - python interface
def julian_ephemeris_millennium(
    datetime_like, apply_correction: Boolean=False) -> NDArray[np.float64]:
    cdef double[:] ut, delta_t
    ut, delta_t = unixtime_delta_t(datetime_like, apply_correction)
    return np.asfarray(_julian_ephemeris_millennium(ut, delta_t))


# =============================================================================
# POLYNOMIAL EXPRESSIONS FOR DELTA
# =============================================================================
cdef double[:] _pe4dt(
    long[:] years, long[:] months, bint apply_corection) noexcept:
    cdef int n, i
    cdef long year, month
    cdef double[:] delta_t

    n = len(years)
    delta_t = lib.view1d(n)

    for i in prange(n, nogil=True):
        year = years[i]
        month = months[i]
        delta_t[i] = lib.pe4dt(year, month, apply_corection)

    return delta_t

# - python interface
def pe4dt(datetime_like, apply_correction:Boolean=False):
    cdef np.ndarray dt
    cdef long[:] y, m

    dt = lib.dtarray(datetime_like)
    y = lib.years(dt)
    m = lib.months(dt)

    return np.asfarray(_pe4dt(y, m, apply_correction))


# =============================================================================
# 
# =============================================================================
cdef double[:] _radius_vector(double[:] unixtime, double[:] delta_t, int num_threads = 1) noexcept:
    cdef int n, i
    cdef double[:] jme, out

    jme = _julian_ephemeris_millennium(unixtime, delta_t)
    n = len(jme)
    out = np.zeros_like(jme)

    for i in prange(n, nogil=True, num_threads=num_threads):
        # out[i] = lib.longitude_latitude_and_radius_vector(jme[i])[2]
        out[i] = terms.radius_vector(jme[i], num_threads=num_threads)

    return out

# - python interface
def radius_vector(datetime_like, apply_correction=False):
    cdef double[:] ut, delta_t

    ut, delta_t = unixtime_delta_t(datetime_like, apply_correction)

    return np.asarray(_radius_vector(ut, delta_t))




cdef double[:, :] _time_components(
    double[:, :] out,  double[:] unixtime, double[:] delta_t, int num_threads = 1
) noexcept nogil: # type: ignore
    cdef int n, i
    cdef double ut, dt, jd, jc, jce, jme, L, B, R, O, delta_psi, delta_eps, E, delta_tau, Lambda

    n = len(unixtime)

    for i in prange(n, nogil=True, num_threads=num_threads):
        ut  = unixtime[i]
        dt = delta_t[i]

        # - 3.1. Calculate the Julian and Julian Ephemeris Day, Century, and Millennium:
        jd = lib.julian_day(ut)
        jc = lib.julian_century(jd)
        # jde = lib.julian_ephemeris_day(jd, dt)
        jce = lib.julian_ephemeris_century(lib.julian_ephemeris_day(jd, dt))
        jme = lib.julian_ephemeris_millennium(jce)

        # - 3.2 Calculate the Earth heliocentric longitude, latitude, 
        # and radius vector (L, B, and R)
        L = terms.longitude(jme, num_threads=num_threads)                                   # L = ∑ X j *Yi, j
        B = terms.latitude(jme, num_threads=num_threads)                                    # B = ∑ X j *Yi, j
        R = terms.radius_vector(jme, num_threads=num_threads)                                # R = ∑ X j *Yi, j
        # L, B, R = lib.longitude_latitude_and_radius_vector(jme)

        # - 3.3 Calculate the geocentric longitude and latitude
        O = (L + 180.0) % 360.0                                                 # Θ = L + 180 geocentric longitude (in degrees)

        # - 3.4 Calculate the nutation in longitude and obliquity
        (
            delta_psi, delta_eps                                                # ∆ψ, ∆ε
        ) = terms.nutation_in_longitude_and_obliquity(jce, num_threads=num_threads)
        # ∆ψ = (ai + bi * JCE ) *sin( ∑ X j *Yi, j )
        # ∆ε = (ci + di * JCE ) *cos( ∑ X j *Yi, j )
        # - 3.5 Calculate the true obliquity of the ecliptic
        E = lib.true_obliquity_of_the_ecliptic(jme, delta_eps)                  # ε = ε0 / 3600 + ∆ε

        # - 3.6 Calculate the aberration correction (in degrees)
        delta_tau = -20.4898 / (R * 3600.0)                                     # ∆τ = − 26.4898 / 3600 * R 

        # - 3.7 Calculate the apparent sun longitude (in degrees)
        Lambda = (O + delta_psi + delta_tau) % 360.0                            # λ = Θ + ∆ψ + ∆τ

        # - 3.8. Calculate the apparent sidereal time at Greenwich (in degrees) 
        out[APARENT_SIDEREAL_TIME, i] = (                                       # ν = ν0 + ∆ψ * cos ε
            lib.apparent_sidereal_time_at_greenwich(jd, jc, E, delta_psi)
        )
        
        # - 3.9,3.10 Calculate the geocentric sun right ascension & declination
        # out[HELIOCENTRIC_RADIUS_VECTOR, i] = R
        (
            out[TOPOCENTRIC_RIGHT_ASCENSION, i],                                # α = ArcTan2(sin λ *cos ε − tan β *sin ε, cos λ)
            out[TOPOCENTRIC_DECLINATION, i]                                     # δ = Arcsin(sin β *cos ε + cos β *sin ε *sin λ) 
        ) = lib.geocentric_right_ascension_and_declination(Lambda, -B, E)

        # 3.12.1. Calculate the equatorial horizontal parallax of the sun
        # NOTE: in the name of compute this function is performed out of order
        # because it is independent of the spatial components
        out[EQUATOIRAL_HORIZONAL_PARALAX, i] = (                                # ξ = 8.794 / (3600 * R)
            lib.equatorial_horizontal_parallax(R)
        )
    return out


cdef type


cdef double[:,:] time_components(
    object datetime_like,
    bint apply_correction = False, 
    int num_threads = 1
):
    cdef double[:] ut, delta_t
    cdef double[:, :] tc_out
    ut, delta_t = unixtime_delta_t(datetime_like, apply_correction)
    tc_out = lib.view2d(NUM_TIME_COMPONENTS, len(ut))
    _time_components(tc_out, ut, delta_t, num_threads=num_threads)
    return tc_out

def get_time_components(datetime_like: ArrayLike, apply_correction = False, int num_threads = 1):
    return np.asfarray(time_components(datetime_like, apply_correction, num_threads))

cdef void _fast_spa(
    unsigned long[:, :]  indicies,
    double[:,:] tc,
    double[:, :] latitude,               # Y
    double[:, :] longitude,              # X
    double[:, :] elevation,              # Z
    double[:, :] pressure,
    double[:, :] temperature,
    double[:, :] refraction,
    double[:, :, :, :] out,
    int num_threads = 1,
) noexcept nogil: # type: ignore
    cdef unsigned long T, Y, X
    cdef int i, n
    cdef double v, xi, delta, alpha                                             # time components
    cdef double E, lat, lon, pres, temp, refct                                  # spatial components
    cdef double H, H_p, delta_alpha, delta_p, e0, e, theta, theta0, gamma, phi

    n = len(indicies)

    for i in prange(n, nogil=True, num_threads=num_threads):
        # - indicies
        T = indicies[i, 0]
        Y = indicies[i, 1]
        X = indicies[i, 2]

        # ---------------------------------------------------------------------
        # - unpack the time components
        # R       = tc[HELIOCENTRIC_RADIUS_VECTOR, T]                             # R
        v       = tc[APARENT_SIDEREAL_TIME, T]                                  # ν
        xi      = tc[EQUATOIRAL_HORIZONAL_PARALAX, T]                           # ξ
        delta   = tc[TOPOCENTRIC_DECLINATION, T]                                # δ
        alpha   = tc[TOPOCENTRIC_RIGHT_ASCENSION, T]                            # α

        # ---------------------------------------------------------------------
        # - unpack the spatial components
        E       = elevation[Y, X]                                               # h
        lat     = latitude[Y, X]                                                # φ
        lon     = longitude[Y, X]                                               # σ
        pres    = pressure[Y, X]
        temp    = temperature[Y, X]
        refct   = refraction[Y, X]

        # ---------------------------------------------------------------------
        
        # - 3.11. Calculate the observer local hour angle, H (in degrees):
        H = (v + lon - alpha) % 360                                                   # H = ν + σ − α
        

        # 3.12. Calculate the topocentric sun right ascension "’ (in degrees)
        (
            delta_alpha, delta_p
        ) = lib.topocentric_parallax_right_ascension_and_declination(delta, H, E, lat, xi)

        # 3.13. Calculate the topocentric local hour angle, H’ (in degrees)
        H_p = H - delta_alpha                                               # H' = H − ∆α

        # 3.14. Calculate the topocentric zenith angle
        (
            e, e0, theta, theta0                                                               # e, e0
        ) = lib.topocentric_azimuth_angle(lat, delta_p, H_p, pres, temp,  refct)
        
        # 3.15.	 Calculate the topocentric azimuth angle
        gamma = lib.topocentric_astronomers_azimuth(H_p, delta_p, lat)
        phi = (gamma + 180) % 360                                               # φ = γ + 180

        out[0, T, Y, X] = theta
        out[1, T, Y, X] = theta0
        out[2, T, Y, X] = e
        out[3, T, Y, X] = e0
        out[4, T, Y, X] = phi

cdef void _fast_spa2(
    (int,int,int) shape,
    double[:,:] tc,
    double[:, :] latitude,               # Y
    double[:, :] longitude,              # X
    double[:, :] elevation,              # Z
    double[:, :] pressure,
    double[:, :] temperature,
    double[:, :] refraction,
    double[:, :, :, :] out,
    int num_threads = 1,
) noexcept nogil: # type: ignore
    cdef unsigned long T, Y, X
    cdef int num_t, num_y, num_x
    cdef double v, xi, delta, alpha                                             # time components
    cdef double E, lat, lon, pres, temp, refct                                  # spatial components
    cdef double H, H_p, delta_alpha, delta_p, e0, e, theta, theta0, gamma, phi

    num_t = shape[0]
    num_y = shape[1]
    num_x = shape[2]
    for T in prange(num_t, nogil=False, num_threads=num_threads):
        # - unpack the time components
        v       = tc[APARENT_SIDEREAL_TIME, T]                                  # ν
        xi      = tc[EQUATOIRAL_HORIZONAL_PARALAX, T]                           # ξ
        delta   = tc[TOPOCENTRIC_DECLINATION, T]                                # δ
        alpha   = tc[TOPOCENTRIC_RIGHT_ASCENSION, T]                            # α
        for Y in prange(num_y, nogil=False, num_threads=num_threads):
            for X in prange(num_x, nogil=False, num_threads=num_threads):
            # ---------------------------------------------------------------------
                # - unpack the spatial components
                E       = elevation[Y, X]                                               # h
                lat     = latitude[Y, X]                                                # φ
                lon     = longitude[Y, X]                                               # σ
                pres    = pressure[Y, X]
                temp    = temperature[Y, X]
                refct   = refraction[Y, X]

                # ---------------------------------------------------------------------
                
                # - 3.11. Calculate the observer local hour angle, H (in degrees):
                H = (v + lon - alpha) % 360                                                   # H = ν + σ − α
                

                # 3.12. Calculate the topocentric sun right ascension "’ (in degrees)
                (
                    delta_alpha, delta_p
                ) = lib.topocentric_parallax_right_ascension_and_declination(delta, H, E, lat, xi)

                # 3.13. Calculate the topocentric local hour angle, H’ (in degrees)
                H_p = H - delta_alpha                                               # H' = H − ∆α

                # 3.14. Calculate the topocentric zenith angle
                (
                    e, e0, theta, theta0                                                               # e, e0
                ) = lib.topocentric_azimuth_angle(lat, delta_p, H_p, pres, temp,  refct)
                
                # 3.15.	 Calculate the topocentric azimuth angle
                gamma = lib.topocentric_astronomers_azimuth(H_p, delta_p, lat)
                phi = (gamma + 180) % 360                                               # φ = γ + 180

                out[0, T, Y, X] = theta
                out[1, T, Y, X] = theta0
                out[2, T, Y, X] = e
                out[3, T, Y, X] = e0
                out[4, T, Y, X] = phi



cdef double[:, :] _grid_component(
    x: ArrayLike | None,
    np.ndarray like,
    float fill_value
):
    cdef double[:,:] out
    if x is None:
        x = np.full_like(like, fill_value) # type: ignore
    elif np.isscalar(x):
        x = np.full_like(like, x) # type: ignore
    else:
        x = np.asfarray(x, dtype=np.float64)

    # x = np.asfarray(x, dtype=np.float64)
    if x.ndim == 1:
        out, _ = np.meshgrid(x, x) # type: ignore
    else:
        out = x
    

    
    return out


def fast_spa(
    datetime_like: ArrayLike,
    latitude: ArrayLike,
    longitude: ArrayLike,
    elevation: ArrayLike | None = None,
    pressure: ArrayLike | None = None,
    temperature: ArrayLike | None = None,
    refraction: ArrayLike | None = None,
    apply_correction = False,
    int num_threads = 1,
):
    cdef double[:] ut, delta_t
    ut, delta_t = unixtime_delta_t(datetime_like, apply_correction)

    latitude = np.asfarray(latitude, dtype=np.float64)
    longitude = np.asfarray(longitude, dtype=np.float64)
    assert latitude.ndim == longitude.ndim
    input_dim = latitude.ndim

    if input_dim == 1:
        latitude, longitude = np.meshgrid(latitude, longitude)
    assert latitude.ndim == longitude.ndim == 2
    shape = (len(ut), len(latitude), len(longitude))
    cdef double[:, :] tc_out
    cdef double[:, :, :, :] out

    # -time components
    tc_out = lib.view2d(NUM_TIME_COMPONENTS, shape[0])
    _time_components(tc_out, ut, delta_t, num_threads=num_threads)

    
    out = lib.view4d(5, shape[0], shape[1], shape[2])
    # indicies = _idxarray(shape)

    # shape = shape[0], shape[1], shape[2]
    # _fast_spa(
    #     indicies, 
    #     tc_out,
    #     latitude, # type: ignore
    #     longitude, # type: ignore
    #     _grid_component(elevation, latitude, 0.0), 
    #     _grid_component(pressure, latitude, 0.0), 
    #     _grid_component(temperature, latitude, 0.0),
    #     _grid_component(refraction, latitude, 0.5667),
    #     out,
    #     num_threads=num_threads,
    # )
    _fast_spa2(
        shape, 
        tc_out,
        latitude, # type: ignore
        longitude, # type: ignore
        _grid_component(elevation, latitude, 0.0), 
        _grid_component(pressure, latitude, 0.0), 
        _grid_component(temperature, latitude, 0.0),
        _grid_component(refraction, latitude, 0.5667),
        out,
        num_threads=num_threads,
    )
    x = np.asfarray(out)
    
    if input_dim == 1:
        # squeeze out the x, mesh grid
        x = x[..., 0]
    return x
