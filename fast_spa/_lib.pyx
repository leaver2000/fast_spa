# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

# pyright: reportGeneralTypeIssues=false
cimport cython
import cython

from libc.math cimport sin, cos, atan2, tan



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double pe4dt(
    long year, long month, bint apply_corection
) noexcept nogil: # type: ignore
    """ref https://eclipse.gsfc.nasa.gov/SEcat5/deltatpoly.html"""
    cdef double delta_t, u, y, t

    y = year + (month - 0.5) / 12

    if year < -500:
        u = (y - 1820) / 100
        delta_t = -20 + 32 * u**2
    elif year < 500:
        u = y / 100
        delta_t = (
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
        delta_t = (
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
        delta_t = 120 - 0.9808 * t - 0.01532 * t**2 + t**3 / 7129
    elif year < 1800:
        t = y - 1700
        delta_t = (
            8.83
            + 0.1603 * t
            - 0.0059285 * t**2
            + 0.00013336 * t**3
            - t**4 / 1174000
        )
    elif year < 1860:
        t = y - 1800
        delta_t = (
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
        delta_t = (
            7.62
            + 0.5737 * t
            - 0.251754 * t**2
            + 0.01680668 * t**3
            - 0.0004473624 * t**4
            + t**5 / 233174
        )
    elif year < 1920:
        t = y - 1900
        delta_t = (
            -2.79
            + 1.494119 * t
            - 0.0598939 * t**2
            + 0.0061966 * t**3
            - 0.000197 * t**4
        )
    elif year < 1941:
        t = y - 1920
        delta_t = 21.20 + 0.84493 * t - 0.076100 * t**2 + 0.0020936 * t**3
    elif year < 1961:
        t = y - 1950
        delta_t = 29.07 + 0.407 * t - t**2 / 233 + t**3 / 2547
    elif year < 1986:
        t = y - 1975
        delta_t = 45.45 + 1.067 * t - t**2 / 260 - t**3 / 718
    elif year < 2005:
        t = y - 2000
        delta_t = (
            63.86
            + 0.3345 * t
            - 0.060374 * t**2
            + 0.0017275 * t**3
            + 0.000651814 * t**4
            + 0.00002373599 * t**5
        )
    elif year < 2050:
        t = y - 2000
        delta_t = 62.92 + 0.32217 * t + 0.005589 * t**2
    elif year < 2150:
        delta_t = -20 + 32 * ((y - 1820) / 100) ** 2 - 0.5628 * (2150 - y)
    else:
        u = (y - 1820) / 100
        delta_t = -20 + 32 * u**2

    if apply_corection:
        delta_t -= -0.000012932 * (y - 1955) ** 2

    return delta_t                                                              # ΔT

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double apparent_sidereal_time_at_greenwich(
    double jd, double jc, double E, double delta_psi
) noexcept nogil: # type: ignore
    cdef double v0, v

    v0 = (
        280.46061837 
        + 360.98564736629 * (jd - 2451545) 
        + 0.000387933 * jc **2 
        - jc**3 / 38710000
    ) % 360

    v = v0 + delta_psi * cos(radians(E))                                             # ν = ν0 + ∆ψ * cos ε

    return v

# =====================================================================================================================
# 3.5. Calculate the true obliquity of the ecliptic, g (in degrees):
# =====================================================================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
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
@cython.nonecheck(False)
@cython.cdivision(True)
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
