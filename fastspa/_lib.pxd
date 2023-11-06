# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

# pyright: reportGeneralTypeIssues=false
cimport cython
from libc.math cimport (
    sin, cos, sqrt, atan2, asin, acos, fabs, fmod, floor, ceil, tan, pi
)
cimport numpy as cnp
import numpy as np

# ctypedef double double
ctypedef unsigned long long u64

cdef double[:, :] L0, L1, L2, L3, L4, L5, B0, B1, B2, B3, B4, B5, R1, R2, R3, R4, R5
# - math
    

# cdef double deg2rad(double deg) nogil
# cdef double rad2deg(double rad) nogil

# cdef inline double rad2deg(double rad) noexcept nogil: # type: ignore
#     return rad * 180 / pi

# cdef inline double deg2rad(double deg) noexcept nogil: # type: ignore
#     return deg * pi / 180
cdef inline double deg2rad(double deg) noexcept nogil: # type: ignore
    return deg * (pi / 180)

cdef inline double radians(double deg) noexcept nogil: # type: ignore
    return deg * (pi / 180)

cdef inline  double rad2deg(double rad) noexcept nogil: # type: ignore
    return rad * 180 / pi

cdef inline double arctan(double x) noexcept nogil: # type: ignore
    return atan2(x, 1)
# =============================================================================
# - time
# =============================================================================
cdef int UNIX_EPOCH_0 = 1970  # Unix epoch start year
cdef double DAYS_PER_YEAR = 365.25 # days per year
cdef double SECONDS_PER_YEAR = 31536000 # seconds per year
cdef double SECONDS_PER_MONTH = 2629743.83 # seconds per month


cdef inline double m360(double x) nogil:
    return x % 360

cdef inline double julian_day(double ut) nogil:
    return ut / 86400 + 2440587.5

cdef inline double julian_century(double jd) nogil:
    return (jd - 2451545) / 36525

cdef inline double julian_ephemeris_day(double jd, double delta_t) nogil:
    return jd + delta_t / 86400

cdef inline double julian_ephemeris_century(double jde) nogil:
    return (jde - 2451545) / 36525

cdef inline double julian_ephemeris_millennium(double jce) nogil:
    return jce / 10

cdef double[:]polynomial_expression_for_delta_t(
    long[:] years, long[:] months, bint apply_corection
)

cdef inline double apparent_sidereal_time_at_greenwich(
    double jd, double jc, double E
) noexcept nogil: # type: ignore
    cdef double v0, v
    v0 = (280.46061837 + 360.98564736629 * (jd - 2451545) + 0.000387933 * jc **2 - jc**3 / 38710000) % 360
    v = v0 + E * cos(deg2rad(E))
    return v

# =============================================================================
cdef inline double earth_periodic_term_summation(
    double a, double b, double c, double jme
) nogil:
    """3.2.	 Calculate the Earth heliocentric longitude, latitude, and radius 
    vector (L, B, # and R): """
    return a * cos(b + c * jme)

cdef tuple[double, double, double] heliocentric_longitude_latitude_and_radius_vector(double JME) nogil             # 3.3             
cdef tuple[double, double]nutation_in_longitude_and_obliquity(double JCE) nogil # 3.4
cdef double true_obliquity_of_the_ecliptic(double JME, double delta_eps) nogil  # 3.5
cdef tuple[double, double] right_ascension_and_declination(                     # 3.9
    double apparent_lon, double geocentric_lat, double true_ecliptic_obliquity
) nogil
# =============================================================================
cdef inline double equatorial_horizontal_parallax(double earth_radius_vector) nogil:
    cdef double xi 
    xi = 8.794 / (3600 * earth_radius_vector)
    return xi


cdef inline double uterm(double observer_latitude) nogil:
    cdef double u
    u = arctan(0.99664719 * tan(radians(observer_latitude)))
    return u


cdef inline double xterm(double u, double observer_latitude, double observer_elevation) nogil:
    cdef double x
    x = cos(u) + observer_elevation / 6378140 * cos(radians(observer_latitude))
    return x

cdef inline double yterm(double u, double observer_latitude, double observer_elevation) nogil:
    cdef double y
    y = 0.99664719 * sin(u) + observer_elevation / 6378140 * sin(radians(observer_latitude))
    return y
