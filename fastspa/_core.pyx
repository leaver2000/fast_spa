
import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange
from ._lib cimport (
    polynomial_expression_for_delta_t,
    julian_day, 
    julian_century, 
    julian_ephemeris_day, 
    julian_ephemeris_century,
    julian_ephemeris_millennium,
    apparent_sidereal_time_at_greenwich,
    nutation_in_longitude_and_obliquity,
    heliocentric_longitude_latitude_and_radius_vector,
    true_obliquity_of_the_ecliptic,
    right_ascension_and_declination,
    equatorial_horizontal_parallax,
    uterm, 
    xterm,
    yterm
)

from fastspa import _utils
cnp.import_array()
ctypedef unsigned long long u64




@cython.boundscheck(False)
@cython.boundscheck(False)
cdef double[:,:] get_time_components(double[:] unixtime, double[:] delta_t) noexcept:
    cdef int n, i
    cdef double ut, dt, jd, jc, jde, jce, jme, L, B, R, O, DeltaPSI, DeltaE, E, DeltaT, Lambda, alpha, delta, v, v0
    cdef double[:, :] out
    n = len(unixtime)
    out = np.zeros((4, n), dtype=np.float64) # type: ignore

    for i in prange(n, nogil=True):
        ut  = unixtime[i]
        dt = delta_t[i]
        jd = julian_day(ut)
        jc = julian_century(jd)
        jde = julian_ephemeris_day(ut, dt)
        jce = julian_ephemeris_century(jde)
        jme = julian_ephemeris_millennium(jce)

        # 3.2.	 Calculate the Earth heliocentric longitude, latitude, and radius vector (L, B, and R): 
        L, B, R = heliocentric_longitude_latitude_and_radius_vector(jme)

        # 3.3.1. Calculate the geocentric longitude (in degrees)
        O = (L + 180.0) % 360.0                                                 # Θ = L + 180 

        # 3.4. Calculate the nutation in longitude and obliquity
        (
            DeltaPSI,                                                           # ∆ψ = (ai + bi * JCE ) *sin( ∑ X j *Yi, j )
            DeltaE                                                              # ∆ε = (ci + di * JCE ) *cos( ∑ X j *Yi, j )
        ) = nutation_in_longitude_and_obliquity(jce)
        
        # 3.5. Calculate the true obliquity of the ecliptic
        E = true_obliquity_of_the_ecliptic(jme, DeltaE)                         # ε = ε0 / 3600 + ∆ε

        # 3.6. Calculate the aberration correction (in degrees)
        DeltaT = -20.4898 / (R * 3600.0)                                        # ∆τ = − 26.4898 / 3600 * R 
        # 3.7. Calculate the apparent sun longitude (in degrees)
        Lambda = (O + DeltaPSI + DeltaT) % 360.0                                     # λ = Θ + ∆ψ + ∆τ
        # 3.9.	 Calculate the geocentric sun right ascension, " (in degrees): 
        # 3.10.	 Calculate the geocentric sun declination, * (in degrees):
        # geocentric sun declination
        alpha, delta = right_ascension_and_declination(
            Lambda,
            -B, # geocentric latitude
            E,
        )
        
        out[0, i] = R
        out[1, i] = alpha
        out[2, i] = delta
        # ====================================================================================================
        # 3.8. Calculate the apparent sidereal time at Greenwich at any given time, < (in degrees): 
        # v0 = (280.46061837 + 360.98564736629 * (jd - 2451545) + 0.000387933 * jc **2 - jc**3 / 38710000) % 360
        out[3, i] = apparent_sidereal_time_at_greenwich(jd, jc, E)

        
        
    return out
import itertools


@cython.boundscheck(False)
cdef u64[:,:] _idxarray(tuple[u64, u64, u64, u64] shape) noexcept:
    cdef u64 num_t, num_z, num_y, num_x
    cdef u64[:, :] indicies_ 
    num_t, num_z, num_y, num_x = shape
    indicies_ = np.array(
        list(itertools.product(range(num_t), range(num_z), range(num_y), range(num_x))),
        dtype=np.uint64,
    ) # type: ignore
    return indicies_


cdef fast_spa(
    tuple[u64, u64, u64, u64] shape,    # (T, Z, Y, X)
    double[:] unixtime,                 # T
    double[:] delta_t,                  # T
    double[:] elevation,                # Z
    double[:,:] latitude,               # Y
    double[:,:] longitude,              # X
):
    cdef u64 T, Z, Y, X, i
    cdef double R, alpha, delta, v, elv, lat, lon, xi, u, x, y, H 
    cdef double[:,:] tc
    cdef u64[:,:] indicies_ = _idxarray(shape)
    cdef int nidx = len(indicies_)
    cdef double[:,:,:,:,:] out = np.zeros((1,) + shape, dtype=np.float64) # type: ignore (C, T, Z, Y, X) 

    # - The time components are independent of the spatial components
    # so they are computed prior to the loop.
    tc = get_time_components(unixtime, delta_t) #((R, alpha, delta), T)

    for i in prange(nidx, nogil=True):
        # - indicies
        T = indicies_[i, 0]
        Z = indicies_[i, 1]
        Y = indicies_[i, 2]
        X = indicies_[i, 3]
        # ---------------------------------------------------------------------
        # - vertical (Z)
        elv = elevation[Z]                                                      #  h
        # - spatial (Y, X)
        lat = latitude[Y, X]                                                    #  φ
        lon = longitude[Y, X]                                                   #  σ
        # ---------------------------------------------------------------------
        # - unpack the time components
        R, alpha, delta, v = tc[0, T], tc[1, T], tc[2, T], tc[3, T]             # R, α, δ, ν
        # 3.11. Calculate the observer local hour angle, H (in degrees):
        H = (v + lon + alpha) % 360                                             # H = ν + σ − α
        # 3.12. Calculate the topocentric sun right ascension "’ (in degrees): 
        xi = equatorial_horizontal_parallax(R)
        u = uterm(lat)                                                          # u       
        x = xterm(u, lat, elv)                                                  # x
        y = yterm(u, lat, elv)                                                  # y






        out[0, T, Z, Y, X] = R


    return out



def main(
    datetime_like
):
    from fastspa import _utils
    dt = np.asanyarray(datetime_like, dtype="datetime64[ns]").ravel()
    ut = dt.astype(np.float64) // 1e9
    year = dt.astype("datetime64[Y]").astype(int) + 1970
    month = dt.astype("datetime64[M]").astype(int) % 12 + 1
    delta_t = polynomial_expression_for_delta_t(year, month, False)

    assert np.allclose(
        delta_t,
        _utils.calculate_deltat(
            dt.astype("datetime64[Y]").astype(int) + 1970,
            dt.astype("datetime64[M]").astype(int) % 12 + 1,
        )
    )
    z = np.array([0.0])
    x = np.linspace(-180, 180, 20)
    y = np.linspace(-90, 90, 20)
    xx,yy = np.meshgrid(x, y)
    shape = (len(dt), len(z), len(y), len(x))
    
    x = fast_spa(
        shape,
        ut,
        delta_t,
        z,
        
        yy,
        xx,
    )
    print(np.asarray(x))
# ====================================================================================================

cdef double[:,:] test_get_time_components(double[:] unixtime, double[:] delta_t):
    cdef int n, i
    cdef double ut,dt,jd,jc,jde,jce,jme,L,B,R,O,DeltaPSI,DeltaE,E,DeltaT,Lambda,alpha,delta
    cdef double[:,:] out
    n = len(unixtime)
    out = np.zeros((3, n), dtype=np.float64)
    for i in range(n):
        ut  = unixtime[i]
        dt = delta_t[i]
        jd = julian_day(ut)
        assert jd == _utils.julian_day(ut)
        
        jc = julian_century(jd)
        assert jc == _utils.julian_century(jd)

        jde = julian_ephemeris_day(ut, dt)
        assert jde == _utils.julian_ephemeris_day(ut, dt)

        jce = julian_ephemeris_century(jde)
        assert jce == _utils.julian_ephemeris_century(jde)

        jme = julian_ephemeris_millennium(jce)
        assert jme == _utils.julian_ephemeris_millennium(jce)
        

        # 3.2.	 Calculate the Earth heliocentric longitude, latitude, and radius vector (L, B, and R): 
        L, B, R = heliocentric_longitude_latitude_and_radius_vector(jme)
        

        assert np.allclose(
            (L, B, R), _utils.heliocentric_longitude_latitude_and_radius_vector(jme)
        )
        # 3.3.1. Calculate the geocentric longitude (in degrees)
        O = (L + 180.0) % 360.0                                                 # Θ = L + 180 


        # 3.4. Calculate the nutation in longitude and obliquity
        (
            DeltaPSI,                                                           # ∆ψ = (ai + bi * JCE ) *sin( ∑ X j *Yi, j )
            DeltaE                                                              # ∆ε = (ci + di * JCE ) *cos( ∑ X j *Yi, j )
        ) = nutation_in_longitude_and_obliquity(jce)
        
        # 3.5. Calculate the true obliquity of the ecliptic
        E = true_obliquity_of_the_ecliptic(jme, DeltaE)                         # ε = ε0 / 3600 + ∆ε
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
        alpha, delta = right_ascension_and_declination(
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
        # ====================================================================================================
        # SKIP: 3.11. Calculate the observer local hour angle, H (in degrees)
    return out
