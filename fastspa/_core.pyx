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
from . cimport _lib as lib
cnp.import_array()

    

ctypedef signed long long i64
ctypedef unsigned long long u64

Boolean: typing.TypeAlias = "bool | bint"


# =============================================================================
# datetime64 arary functions
# =============================================================================
cdef cnp.ndarray cast_array(cnp.ndarray a, int n) noexcept:
    return cnp.PyArray_Cast(a, n) # type: ignore


@cython.boundscheck(True)
cdef cnp.ndarray dtarray(datetime_like) noexcept:
    """
    main entry point for datetime_like.
    need to add validation to the object so everthing else can be marked as
    `nogil`
    
    """
    cdef cnp.ndarray dt
    dt = np.asanyarray(datetime_like, dtype="datetime64[ns]").ravel() # type: ignore

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

    dt = dtarray(datetime_like)
    ut = _unixtime(dt)
    y = _years(dt)
    m = _months(dt)
    delta_t = _pe4dt(y, m, apply_correction)

    return ut, delta_t


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
            lib.julian_ephemeris_century(lib.julian_ephemeris_day(
                lib.julian_day(ut), dt))
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

    dt = dtarray(datetime_like)
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
    cdef double ut, dt, jd, jc, jde, jce, jme, L, B, R, O, DeltaPSI, DeltaE, E, DeltaT, Lambda, alpha, delta, v, v0
    cdef double[:, :] out

    n = len(unixtime)
    out = np.zeros((5, n), dtype=np.float64) # type: ignore

    for i in prange(n, nogil=True):
        ut  = unixtime[i]
        dt = delta_t[i]
        jd = lib.julian_day(ut)
        jc = lib.julian_century(jd)
        jde = lib.julian_ephemeris_day(jd, dt)
        jce = lib.julian_ephemeris_century(jde)
        jme = lib.julian_ephemeris_millennium(jce)
        # print(
        #     "JME",
        #     jme, 
        #     spa.julian_ephemeris_millennium(
        #         spa.julian_ephemeris_century(
        #             spa.julian_ephemeris_day(
        #                 spa.julian_day(ut), dt
        #             )
        #         )
        #     )
        # )
        # 3.2	 Calculate the Earth heliocentric longitude, latitude, and radius vector (L, B, and R): 
        L, B, R = lib.longitude_latitude_and_radius_vector(jme)
        # assert np.allclose(
        #     spa.solar_position_numpy(
        #         ut, None, None, elev=E, pressure=None, temp=None, delta_t=dt, numthreads=None, atmos_refract=None, esd=True
        #     )[0], 
        #     R
        # )
        # 3.3 Calculate the geocentric longitude and latitude
        O = (L + 180.0) % 360.0                                                 # Θ = L + 180 geocentric longitude (in degrees)

        # 3.4 Calculate the nutation in longitude and obliquity
        (
            DeltaPSI,                                                           # ∆ψ = (ai + bi * JCE ) *sin( ∑ X j *Yi, j )
            DeltaE                                                              # ∆ε = (ci + di * JCE ) *cos( ∑ X j *Yi, j )
        ) = lib.nutation_in_longitude_and_obliquity(jce)
        
        # 3.5 Calculate the true obliquity of the ecliptic
        E = lib.true_obliquity_of_the_ecliptic(jme, DeltaE)                     # ε = ε0 / 3600 + ∆ε

        # 3.6 Calculate the aberration correction (in degrees)
        DeltaT = -20.4898 / (R * 3600.0)                                        # ∆τ = − 26.4898 / 3600 * R 

        # 3.7 Calculate the apparent sun longitude (in degrees)
        Lambda = (O + DeltaPSI + DeltaT) % 360.0                                # λ = Θ + ∆ψ + ∆τ

        # 3.8. Calculate the apparent sidereal time at Greenwich (in degrees) 
        v = lib.apparent_sidereal_time_at_greenwich(jd, jc, E, DeltaPSI)        # ν = ν0 + ∆ψ * cos ε

        # assert np.allclose(
        #     v, spa.apparent_sidereal_time(spa.mean_sidereal_time(jd, jc), DeltaPSI, E)
        # )
        # 3.9,3.10 Calculate the geocentric sun right ascension & declination
        (
            alpha,                                                              # α = ArcTan2(sin λ *cos ε − tan β *sin ε, cos λ)
            delta,                                                              # δ = Arcsin(sin β *cos ε + cos β *sin ε *sin λ) 
        ) = lib.geocentric_right_ascension_and_declination(Lambda, -B, E)
        
        out[0, i] = R
        out[1, i] = alpha
        out[2, i] = delta
        out[3, i] = v
        out[4, i] = 8.794 / (3600 * R)
    return out

import pvlib.spa as spa
cdef _fast_spa(
    tuple[u64, u64, u64] shape,    # (T, Y, X)
    double[:] unixtime,                 # T
    double[:] delta_t,                  # T
    double[:,:] elevation,              # Z
    double[:,:] latitude,               # Y
    double[:,:] longitude,              # X
    double[:,:] pressure,
    double[:,:] temperature,
    double[:,:] refraction,
) noexcept:
    cdef u64 T, Z, Y, X, i
    cdef double R, alpha, delta, v, elv, lat, lon, xi, u, x, y, H 
    cdef double[:,:] tc
    cdef u64[:,:] indicies_ = _idxarray(shape)
    cdef int nidx = len(indicies_)
    cdef double[:,:,:,:] out = np.zeros((5,) + shape, dtype=np.float64) # type: ignore (C, T, Z, Y, X) 
    # from fastspa import _utils  
    # - The time components are independent of the spatial components
    # so they are computed prior to the loop.
    tc = _time_components(unixtime, delta_t) #((R, alpha, delta), T)

    # for i in prange(nidx, nogil=True):
    for i in range(nidx):
        # - indicies
        T = indicies_[i, 0]
        # Z = indicies_[i, 1]
        Y = indicies_[i, 1]
        X = indicies_[i, 2]
        # ---------------------------------------------------------------------
        E = elevation[Y, X]                                                      #  h
        lat = latitude[Y, X]                                                    #  φ
        lon = longitude[Y, X]                                                   #  σ
        pres = pressure[Y, X]
        temp = temperature[Y, X]
        refct = refraction[Y, X]
        # ---------------------------------------------------------------------
        # - unpack the time components
        R = tc[0, T]                                                            # R
        assert np.allclose(
            spa.solar_position_numpy(
                unixtime[T], lat, lon, elev=E, pressure=pres, temp=temp, delta_t=delta_t[T], numthreads=None, atmos_refract=refct,esd=True
            )[0], 
            R
        )
        alpha = tc[1, T]                                                        # α
        delta_ = tc[2, T]                                                        # δ
        v = tc[3, T]                                                            # ν
        xi = tc[4, T]                                                           # ξ

        v_, alpha_, delta = spa.solar_position_numpy( # type: ignore
            np.array([unixtime[T]]), lat, lon, elev=E, pressure=pres, temp=temp, delta_t=delta_t[T],
            numthreads=None, atmos_refract=refct, sst=True)
#         print(
#             f"""v:{v} {v_}
# alpha:{alpha} {alpha_}
# delta:{delta} {delta_}"""
#             # v_, alpha_, delta_,
#             # v, alpha, delta,

        
#         )

        assert np.allclose(alpha_, alpha)
        # assert np.allclose(delta_, delta)
        assert np.allclose(v_, v)
        # ---------------------------------------------------------------------
        # assert np.allclose(spa.equatorial_horizontal_parallax(R), xi)
        # 3.11. Calculate the observer local hour angle, H (in degrees):
        
        H = (v + lon - alpha)                                               # H = ν + σ − α
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
            delta_, H, E, lat, xi
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

def main(datetime_like=None):
    if datetime_like is None:
        datetime_like = ["2022-01-01"]
    z = np.array([0.0])
    x = np.linspace(-180, 180, 20)
    y = np.linspace(-90, 90, 20)
    xx, yy = np.meshgrid(x, y)
    fast_spa(
        datetime_like,
        z,
        yy,
        xx,
        apply_correction=False,
    )
    test_get_time_components(
        np.array([0.0]),
        np.array([0.0]),
    )
# ====================================================================================================

cdef double[:,:] test_get_time_components(double[:] unixtime, double[:] delta_t):
    from fastspa import _utils
    cdef int n, i
    cdef double ut,dt,jd,jc,jde,jce,jme,L,B,R,O,DeltaPSI,DeltaE,E,DeltaT,Lambda,alpha,delta
    cdef double[:,:] out
    n = len(unixtime)
    out = np.zeros((3, n), dtype=np.float64)
    for i in range(n):
        ut  = unixtime[i]
        dt = delta_t[i]
        jd = lib.julian_day(ut)
        assert jd == _utils.julian_day(ut)
        
        jc = lib.julian_century(jd)
        assert jc == _utils.julian_century(jd)

        jde = lib.julian_ephemeris_day(ut, dt)
        assert jde == _utils.julian_ephemeris_day(ut, dt)

        jce = lib.julian_ephemeris_century(jde)
        assert jce == _utils.julian_ephemeris_century(jde)

        jme = lib.julian_ephemeris_millennium(jce)
        assert jme == _utils.julian_ephemeris_millennium(jce)
        

        # 3.2.	 Calculate the Earth heliocentric longitude, latitude, and radius vector (L, B, and R): 
        L, B, R = lib.longitude_latitude_and_radius_vector(jme)
        

        assert np.allclose(
            (L, B, R), _utils.heliocentric_longitude_latitude_and_radius_vector(jme)
        )
        # 3.3.1. Calculate the geocentric longitude (in degrees)
        O = (L + 180.0) % 360.0                                                 # Θ = L + 180 


        # 3.4. Calculate the nutation in longitude and obliquity
        (
            DeltaPSI,                                                           # ∆ψ = (ai + bi * JCE ) *sin( ∑ X j *Yi, j )
            DeltaE                                                              # ∆ε = (ci + di * JCE ) *cos( ∑ X j *Yi, j )
        ) = lib.nutation_in_longitude_and_obliquity(jce)
        
        # 3.5. Calculate the true obliquity of the ecliptic
        E = lib.true_obliquity_of_the_ecliptic(jme, DeltaE)                         # ε = ε0 / 3600 + ∆ε
        assert np.allclose(
            E, 
            _utils.true_ecliptic_obliquity(
                _utils.mean_ecliptic_obliquity(jme), 
                DeltaE
            ),
        )

        # 3.6. Calculate the aberration correction (in degrees)
        DeltaT = -20.4898 / (3600.0 * R)                                        # ∆τ = − 26.4898 / 3600 * R 
        assert np.allclose(DeltaT, _utils.aberration_correction(R))
        # 3.7. Calculate the apparent sun longitude (in degrees)
        Lambda = (O + DeltaPSI + DeltaT) % 360.0                                     # λ = Θ + ∆ψ + ∆τ
        # SKIP: 3.8. Calculate the apparent sidereal time at Greenwich at any given time, < (in degrees): 
        # 3.9.	 Calculate the geocentric sun right ascension, " (in degrees): 
        # 3.10.	 Calculate the geocentric sun declination, * (in degrees):
        # geocentric sun declination
        alpha, delta = lib.geocentric_right_ascension_and_declination(
            Lambda,
            -B, # geocentric latitude
            E,
        )
        # ====================================================================================================
        # TEST
        # ====================================================================================================
        delta_tau = _utils.aberration_correction(R)
        Theta = _utils.geocentric_longitude(L)
        beta = _utils.geocentric_latitude(B)
        lamd = _utils.apparent_sun_longitude(Theta, DeltaPSI, delta_tau)
        assert np.allclose(Lambda, lamd)
        assert np.allclose(
            [alpha, delta], 
            [
                _utils.geocentric_sun_right_ascension(
                    apparent_sun_longitude=lamd, 
                    true_ecliptic_obliquity=E, 
                    geocentric_latitude=beta,
                ), 
                _utils.geocentric_sun_declination(
                    apparent_sun_longitude=lamd, 
                    true_ecliptic_obliquity=E, 
                    geocentric_latitude=beta
                )
            ]
        )
        out[0, i] = R
        out[1, i] = alpha
        out[2, i] = delta
        # xi = 8.794 / (3600 * earth_radius_vector)
        # ====================================================================================================
        # SKIP: 3.11. Calculate the observer local hour angle, H (in degrees)
    return out
