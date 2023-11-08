# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=False

cimport cython
from cython.parallel cimport prange # type: ignore

import numpy as np
from numpy.typing import NDArray, ArrayLike
cimport numpy as cnp

import itertools
import typing
import pvlib.spa as spa

from . cimport _lib as lib
cnp.import_array()

    

ctypedef signed long long i64
ctypedef unsigned long long u64

Boolean: typing.TypeAlias = "bool | bint"


cdef int HELIOCENTRIC_RADIUS_VECTOR = 0
cdef int TOPOCENTRIC_RIGHT_ASCENSION = 1
cdef int TOPOCENTRIC_DECLINATION = 2
cdef int APARENT_SIDEREAL_TIME = 3
cdef int EQUATOIRAL_HORIZONAL_PARALAX = 4

# =============================================================================
# datetime64 arary functions
# =============================================================================
cdef cnp.ndarray cast_array(cnp.ndarray a, int n) noexcept:
    return cnp.PyArray_Cast(a, n) # type: ignore


cdef cnp.ndarray _dtarray(datetime_like) noexcept:
    """
    main entry point for datetime_like.
    need to add validation to the object so everthing else can be marked as
    `nogil`
    
    """
    cdef cnp.ndarray dt
    dt = np.asanyarray(datetime_like, dtype="datetime64[ns]") # type: ignore

    return dt


cdef double[:] _unixtime(cnp.ndarray dt) noexcept:
    cdef double[:] ut
    ut = cast_array(dt, cnp.NPY_TYPES.NPY_DOUBLE) // 1e9 # type: ignore
    return ut


cdef long[:] _years(cnp.ndarray dt) noexcept:
    cdef long[:] Y = dt.astype("datetime64[Y]").astype(np.int64) + 1970 # type: ignore
    return Y


cdef long[:] _months(cnp.ndarray dt) noexcept:
    cdef long[:] M = dt.astype("datetime64[M]").astype(np.int64) % 12 + 1 # type: ignore
    return M


cdef unixtime_delta_t(
    datetime_like, apply_correction: Boolean = False) noexcept:
    cdef cnp.ndarray dt
    cdef long[:] y, m
    cdef double[:] ut, delta_t

    dt = _dtarray(datetime_like)
    ut = _unixtime(dt)
    y = _years(dt)
    m = _months(dt)
    delta_t = _pe4dt(y, m, apply_correction)

    return ut, delta_t


def faproperty(f) -> property:
    return property(lambda self: np.asfarray(f(self)))

cdef class Julian:
    cdef double[:] _unixtime
    cdef double[:] _delta_t


    def __init__(self, datetime_like, apply_correction: Boolean = False):
        self._unixtime, self._delta_t = unixtime_delta_t(
            datetime_like, apply_correction
        )

    cdef double[:]_map(self, f, double[:] x):
        cdef int n, i
        cdef double[:] out
        n = len(x)
        out = np.zeros((n,), dtype=np.float64) # type: ignore

        for i in range(n):
            out[i] = f(x[i])

        return out
    
    cdef _day(self):
        return self._map(lib.julian_day, self._unixtime)


    cdef _century(self):
        return self._map(lib.julian_century, self._day())

    cdef _ephemeris_day(self):
        cdef int n, i
        cdef double[:] jd, dt, out

        jd = self._day()
        dt = self._delta_t
        n = len(jd)
        out = np.zeros((n,), dtype=np.float64) # type: ignore

        for i in range(n):
            out[i] = lib.julian_ephemeris_day(jd[i], dt[i])
        
        return out
    
    cdef _ephemeris_century(self):
        return self._map(lib.julian_ephemeris_century, self._ephemeris_day())
    
    cdef _ephemeris_millennium(self):
        return self._map(lib.julian_ephemeris_millennium, self._ephemeris_century())

    @faproperty
    def unixtime(self):
        return self._unixtime
    
    @faproperty
    def delta_t(self):
        return self._delta_t

    @faproperty
    def day(self):
        return self._day()
    
    @faproperty
    def century(self):
        return self._century()
    
    @faproperty
    def ephemeris_day(self):
        return self._ephemeris_day()

    @faproperty
    def ephemeris_century(self):
        return self._ephemeris_century()

    @faproperty
    def ephemeris_millennium(self):
        return self._ephemeris_millennium()
    
cdef double[:] _julian_ephemeris_millennium(
    double[:] unixtime, double[:] delta_t) noexcept:
    cdef int n, i
    cdef double ut, dt
    cdef double[:] out

    n = len(unixtime)
    out = np.zeros((n,), dtype=np.float64) # type: ignore

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
    delta_t = np.zeros(n, dtype=np.float64) # type: ignore

    for i in prange(n, nogil=True):
        year = years[i]
        month = months[i]
        delta_t[i] = lib.pe4dt(year, month, apply_corection)

    return delta_t

# - python interface
def pe4dt(datetime_like, apply_correction:Boolean=False):
    cdef cnp.ndarray dt
    cdef long[:] y, m

    dt = _dtarray(datetime_like)
    y = _years(dt)
    m = _months(dt)

    return np.asfarray(_pe4dt(y, m, apply_correction))


# =============================================================================
# 
# =============================================================================
cdef double[:] _radius_vector(double[:] unixtime, double[:] delta_t) noexcept:
    cdef int n, i
    cdef double[:] jme, out

    jme = _julian_ephemeris_millennium(unixtime, delta_t)
    n = len(jme)
    out = np.zeros_like(jme)

    for i in prange(n, nogil=True):
        out[i] = lib.longitude_latitude_and_radius_vector(jme[i])[2]

    return out

# - python interface
def radius_vector(datetime_like, apply_correction=False):
    cdef double[:] ut, delta_t

    ut, delta_t = unixtime_delta_t(datetime_like, apply_correction)

    return np.asarray(_radius_vector(ut, delta_t))

# =============================================================================
cdef u64[:,:] _idxarray(tuple[u64, u64, u64] shape) noexcept:
    cdef u64 T, Y, X
    cdef u64[:, :] indicies
    T, Y, X = shape 
    indicies = np.array(
        list(itertools.product(range(T), range(Y), range(X))),
        dtype=np.uint64,
    ) # type: ignore

    return indicies

cdef double[:,:] _time_components(double[:] unixtime, double[:] delta_t) noexcept:
    cdef int n, i
    cdef double ut, dt, jd, jc, jde, jce, jme, L, B, R, O, Dpsi, DE, E, Dt, Lambda
    cdef double[:, :] out

    n = len(unixtime)
    out = np.zeros((5, n), dtype=np.float64) # type: ignore

    for i in prange(n, nogil=True):
        ut  = unixtime[i]
        dt = delta_t[i]
        # - 3.1. Calculate the Julian and Julian Ephemeris Day, Century, and Millennium:
        jd = lib.julian_day(ut)
        jc = lib.julian_century(jd)
        jde = lib.julian_ephemeris_day(jd, dt)
        jce = lib.julian_ephemeris_century(jde)
        jme = lib.julian_ephemeris_millennium(jce)

        # - 3.2 Calculate the Earth heliocentric longitude, latitude, 
        # and radius vector (L, B, and R)
        L, B, R = lib.longitude_latitude_and_radius_vector(jme)

        # - 3.3 Calculate the geocentric longitude and latitude
        O = (L + 180.0) % 360.0                                                 # Θ = L + 180 geocentric longitude (in degrees)

        # - 3.4 Calculate the nutation in longitude and obliquity
        (
            Dpsi,                                                               # ∆ψ = (ai + bi * JCE ) *sin( ∑ X j *Yi, j )
            DE                                                                  # ∆ε = (ci + di * JCE ) *cos( ∑ X j *Yi, j )
        ) = lib.nutation_in_longitude_and_obliquity(jce)

        # - 3.5 Calculate the true obliquity of the ecliptic
        E = lib.true_obliquity_of_the_ecliptic(jme, DE)                         # ε = ε0 / 3600 + ∆ε

        # - 3.6 Calculate the aberration correction (in degrees)
        Dt = -20.4898 / (R * 3600.0)                                            # ∆τ = − 26.4898 / 3600 * R 

        # - 3.7 Calculate the apparent sun longitude (in degrees)
        Lambda = (O + Dpsi + Dt) % 360.0                                        # λ = Θ + ∆ψ + ∆τ

        # - 3.8. Calculate the apparent sidereal time at Greenwich (in degrees) 
        out[APARENT_SIDEREAL_TIME, i] = (                                       # ν = ν0 + ∆ψ * cos ε
            lib.apparent_sidereal_time_at_greenwich(jd, jc, E, Dpsi)
        )
        
        # - 3.9,3.10 Calculate the geocentric sun right ascension & declination
        out[HELIOCENTRIC_RADIUS_VECTOR, i] = R
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


cdef _fast_spa(
    tuple[u64, u64, u64] shape,         # (T, Y, X)
    double[:] unixtime,                 # T
    double[:] delta_t,                  # T
    double[:,:] elevation,              # Z
    double[:,:] latitude,               # Y
    double[:,:] longitude,              # X
    double[:,:] pressure,
    double[:,:] temperature,
    double[:,:] refraction,
) noexcept:
    cdef u64 T, Y, X, i, n
    cdef double R, v, xi, delta, alpha                                          # time components
    cdef double E, lat, lon, pres, temp, refct                                  # spatial components
    cdef double[:,:] tc
    cdef u64[:,:] indicies
    cdef double[:,:,:,:] out


    # - The time components are independent of the spatial components
    # so they are computed prior to the loop.
    indicies = _idxarray(shape)
    n = len(indicies)
    tc = _time_components(unixtime, delta_t) #((R, alpha, delta), T)
    out = np.zeros((5,) + shape, dtype=np.float64) # type: ignore (C, T, Z, Y, X)

    # for i in prange(nidx, nogil=True):
    for i in range(n):
        # - indicies
        T = indicies[i, 0]
        Y = indicies[i, 1]
        X = indicies[i, 2]
        # ---------------------------------------------------------------------
        # - unpack the time components
        R       = tc[HELIOCENTRIC_RADIUS_VECTOR, T]                             # R
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
        assert np.allclose(
            spa.solar_position_numpy(
                unixtime[T], lat, lon, elev=E, pressure=pres, temp=temp, delta_t=delta_t[T], numthreads=None, atmos_refract=refct,esd=True
            )[0], 
            R
        )


        # ---------------------------------------------------------------------
        # assert np.allclose(spa.equatorial_horizontal_parallax(R), xi)
        # 3.11. Calculate the observer local hour angle, H (in degrees):
        
        H = (v + lon - alpha)                                                   # H = ν + σ − α
        H %= 360
        # assert np.allclose(H, spa.local_hour_angle(v, lon, alpha))
        # 3.12. Calculate the topocentric sun right ascension "’ (in degrees): 
        # - test
        # u = spa.uterm(lat)
        # x = spa.xterm(u, lat, E)
        # y = spa.yterm(u, lat, E)
        # delta_alpha = spa.parallax_sun_right_ascension(x, xi, H, delta)
        # assert np.allclose(
        #     lib.topocentric_declination_hour_angle(
        #         # alpha, 
        #         delta, H, E, lat, xi
        #         ),
        #         (
        #             spa.topocentric_sun_declination(delta, x, y, xi, delta_alpha, H),
        #             spa.topocentric_local_hour_angle(H, delta_alpha)
        #         )
        #     )
        # 3.12. Calculate the topocentric sun right ascension "’ (in degrees): 
        # 3.13. Calculate the topocentric local hour angle, H’ (in degrees), 
        # delta_prime = spa.topocentric_sun_declination(delta, x, y, xi, delta_alpha, H)
        # H_prime = spa.topocentric_local_hour_angle(H, delta_alpha)
        delta_prime, H_prime = lib.topocentric_declination_hour_angle(
            delta, H, E, lat, xi
        )

        e0 = spa.topocentric_elevation_angle_without_atmosphere(lat, delta_prime, H_prime)
        delta_e = spa.atmospheric_refraction_correction(pres, temp, e0, refct)
        e = spa.topocentric_elevation_angle(e0, delta_e)
        theta = spa.topocentric_zenith_angle(e)
        theta0 = spa.topocentric_zenith_angle(e0)
        gamma = spa.topocentric_astronomers_azimuth(H_prime, delta_prime, lat)
        phi = spa.topocentric_azimuth_angle(gamma)
        # spa.solar_position_numpy
        out[0, T, Y, X] = theta
        out[1, T, Y, X] = theta0
        out[2, T, Y, X] = e
        out[3, T, Y, X] = e0
        out[4, T, Y, X] = phi

    return out


cdef cnp.ndarray _resolve_component(
    input_dim,
    x,
    like,
    fill_value
):
    if x is None:
        return np.full_like(like, fill_value)
    x = np.asfarray(x, dtype=np.float64)
    if input_dim == 1:
        x, _ = np.meshgrid(x, x)
    assert x.shape == like.shape
    return x

def fast_spa(
    datetime_like: ArrayLike,
    latitude: ArrayLike,
    longitude: ArrayLike,
    elevation: ArrayLike | None = None,
    pressure: ArrayLike | None = None,
    temperature: ArrayLike | None = None,
    refraction: ArrayLike | None = None,
    apply_correction = False,
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
    elevation = _resolve_component(input_dim, elevation, latitude, 0.0)
    pressure = _resolve_component(input_dim, pressure, latitude, 0.0)
    temperature = _resolve_component(input_dim, temperature, latitude, 0.0)
    refraction = _resolve_component(input_dim, refraction, latitude, 0.5667)

    shape = (len(ut), len(latitude), len(longitude))

    x = np.asfarray(
        _fast_spa(
            shape, ut, delta_t, elevation, latitude, longitude,pressure, temperature, refraction
        )
    )
    
    if input_dim == 1:
        # squeeze out the x, mesh grid
        x = x[..., 0]
    return x
