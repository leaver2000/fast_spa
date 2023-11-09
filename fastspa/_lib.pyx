# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

# pyright: reportGeneralTypeIssues=false
cimport cython
import cython
# from cython.parallel cimport prange
cimport numpy as cnp
from libc.math cimport sin, cos, atan2, asin, tan

import numpy as np
cnp.import_array()


# =====================================================================================================================
# 3.5. Calculate the true obliquity of the ecliptic, g (in degrees):
# =====================================================================================================================
@cython.boundscheck(False)
@cython.boundscheck(False)
cdef double true_obliquity_of_the_ecliptic(
    double jme, double delta_eps
) noexcept nogil: # type: ignore
    cdef double U, E0
    U = jme / 10 # U = jme / 10
    #  ε0 = 84381448 − U − 155 U 2 + . 3 . 4680 93 . 1999 25 U −  4 5 6 7 5138 . U − 249 67 . U − 39 05 . U + 712 . U +  89 10
    E0 = (
        84381.448 - 4680.93 * U 
        - 1.55 * U**2 
        + 1999.25 * U**3 
        - 51.38 * U**4 
        - 249.67 * U**5 
        - 39.05 * U**6 
        + 7.12 * U**7 
        + 27.87 * U**8 
        + 5.79 * U**9 
        + 2.45 * U**10
    )

    E = (
        E0 * 1.0 / 3600 + delta_eps / 3600
    ) # ε = ε0 / 3600 + ∆ε

    return E


# =============================================================================
# 3.9	 Calculate the geocentric sun right ascension, " (in degrees):
# 3.10.	 Calculate the geocentric sun declination, * (in degrees): 
# =============================================================================




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)
cdef (double, double) geocentric_right_ascension_and_declination(
    double apparent_lon, double geocentric_lat, double true_ecliptic_obliquity
) noexcept nogil: # type: ignore
    cdef double A, B, E, alpha, delta

    # - in radians
    A = radians(apparent_lon)                                                   # λ
    B = radians(geocentric_lat)                                                 # β
    E = radians(true_ecliptic_obliquity)                                        # ε

    alpha = (
        atan2(sin(A) * cos(E) - tan(B) * sin(E), cos(A))
    ) # ArcTan2(sin λ *cos ε  − tan β  * sin ε / cos λ)
    
    delta = (
        arcsin(sin(B) * cos(E) + cos(B) * sin(E) * sin(A))
    ) # Arcsin(sin β * cos ε  + cos β  * sin ε  * sin λ)
    # - in degrees
    alpha = degrees(alpha) % 360.0  
    delta = degrees(delta)                                                      # δ

    return alpha, delta


# =============================================================================
# 3.12 Calculate the topocentric sun right ascension and declination, α' and δ'
# 3.13. Calculate the topocentric local hour angle, H’ (in degrees)
# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
cdef tuple[double,double] topocentric_parallax_right_ascension_and_declination(
    double delta,   # δ geocentric sun declination
    double H,       # H local hour angle
    double E,       # E observer elevation
    double lat,     # observer latitude
    double xi,      # ξ equatorial horizontal parallax
) noexcept nogil: # type: ignore
    cdef double u, x, y, Phi, delta_alpha, delta_p
    
    delta = radians(delta)          # δ
    xi = radians(xi)                # ξ
    Phi = radians(lat)              # ϕ

    # - 3.12.2. Calculate the term u (in radians)
    u = (
        arctan(0.99664719 * tan(Phi))
    ) # u = Acrtan(0.99664719 * tan ϕ)

    # - 3.12.3. Calculate the term x,
    x = (
        cos(u) + E / 6378140 * cos(Phi)
    ) # x = cosu + E / 6378140 * cos ϕ

    # - 3.12.4. Calculate the term y
    y = (
        0.99664719 * sin(u) + E / 6378140 * sin(Phi)
    ) # y = 0.99664719 * sin u + E / 6378140 * sin ϕ

    # - 3.12.5. Calculate the parallax in the sun right ascension (in degrees),
    delta_alpha = atan2(
        -x * sin(xi) * sin(H),
        cos(delta) - x * sin(xi) * cos(H)
    ) # ∆α = Arctan2(-x *sin ξ *sin H / cosδ − x * sin ξ * cos H)

    delta_p = asin(
        sin(delta) - y * sin(xi) * cos(delta_alpha)
    ) # ∆' = Arcsin(sinδ − y * sin ξ * cos ∆α)


    return degrees(delta_alpha), degrees(delta_p)


# 3.14. Calculate the topocentric zenith angle,
cdef tuple[double, double, double, double] topocentric_azimuth_angle(
    double phi,         # φ observer latitude
    double delta_p,     # δ’ topocentric sun declination
    double H_p,         # H’ topocentric local hour angle
    double P,           # P is the annual average local pressure (in millibars)
    double T,           # T is the annual average local temperature (in degrees Celsius)
    double refraction
) noexcept nogil: # type: ignore
    cdef double e, e0, delta_e, theta, theta0
    # - in radians
    phi = radians(phi)
    delta_p = radians(delta_p)
    H_p = radians(H_p)

    # 3.14.1
    e0 = (
        arcsin(sin(phi)* sin(delta_p) + cos(phi) * cos(delta_p) * cos(H_p))
    ) # e0 = Arcsin(sin ϕ *sin δ'+ cos ϕ *cos δ'*cos H') 

    # - in degrees
    e0 = degrees(e0)

    # 3.14.2
    delta_e = (
        (P / 1010.0) * (283.0 / (273 + T))  * 1.02 / (60 * tan(radians(e0 + 10.3 / (e0 + 5.11))))
    ) # ∆e = (P / 1010) * (283 / (273 + T)) * 1.02 / (60 * tan(radians(e0 + 10.3 / (e0 + 5.11))))
    # Note that ∆e = 0 when the sun is below the horizon.
    delta_e *= e0 >= -1.0 * (0.26667 + refraction)

    # 3.14.3. Calculate the topocentric elevation angle, e (in degrees), 
    e =  e0 + delta_e       # e = e0 + ∆e

    # 3.14.4. Calculate the topocentric zenith angle, 2 (in degrees),
    theta = 90 - e          # θ = 90 − e
    theta0 = 90 - e0        # θ0 = 90 − e0

    return e, e0, theta, theta0 



cdef double topocentric_astronomers_azimuth(
    double topocentric_local_hour_angle,
    double topocentric_sun_declination,
    double observer_latitude
) noexcept nogil: # type: ignore
    cdef double topocentric_local_hour_angle_rad, observer_latitude_rad, gamma
    topocentric_local_hour_angle_rad = radians(topocentric_local_hour_angle)
    observer_latitude_rad = radians(observer_latitude)
    num = sin(topocentric_local_hour_angle_rad)
    denom = (cos(topocentric_local_hour_angle_rad)
             * sin(observer_latitude_rad)
             - tan(radians(topocentric_sun_declination))
             * cos(observer_latitude_rad))
    gamma = degrees(arctan2(num, denom))
    return gamma % 360

