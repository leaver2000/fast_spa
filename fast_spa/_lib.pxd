# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

# pyright: reportGeneralTypeIssues=false
import cython
cimport cython
from cython.view cimport array as cvarray

from libc.math cimport atan2, asin, pi

cimport numpy as cnp
import numpy as np


# =============================================================================
# - memoryview
# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline cnp.ndarray cast_array(cnp.ndarray a, int n) noexcept:
    return cnp.PyArray_Cast(a, n) # type: ignore

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline viewnd(tuple shape) noexcept:
    return cvarray(shape, itemsize=8, format='d')

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double[:] view1d(int a) noexcept:
    cdef  double[:] out = viewnd((a,))
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double[:, :] view2d(int a, int b) noexcept:
    cdef  double[:, :] out = viewnd((a, b))
    return out
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double[:, :, :] view3d(int a, int b, int c) noexcept:
    cdef  double[:, :, :] out = viewnd((a, b, c))
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double[:, :, :, :] view4d(int a, int b, int c, int d) noexcept:
    cdef  double[:, :, :, :] out = viewnd((a, b, c, d))
    return out

# =============================================================================
# - math
# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double radians(double deg) noexcept nogil: # type: ignore
    return deg * (pi / 180)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double degrees(double rad) noexcept nogil: # type: ignore
    return (rad * 180) / pi

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double arctan(double x) noexcept nogil: # type: ignore
    return atan2(x, 1)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double arcsin(double x) noexcept nogil: # type: ignore
    return asin(x)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double arctan2(double y, double x) noexcept nogil: # type: ignore
    return atan2(y, x)

# =============================================================================
# - time
# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline cnp.ndarray dtarray(datetime_like) noexcept:
    """
    main entry point for datetime_like.
    need to add validation to the object so everthing else can be marked as
    `nogil`
    
    """
    cdef cnp.ndarray dt
    dt = np.asanyarray(datetime_like, dtype="datetime64[ns]") # type: ignore

    return dt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double[:] unixtime(cnp.ndarray dt) noexcept:
    cdef double[:] ut
    ut = cast_array(dt, cnp.NPY_TYPES.NPY_DOUBLE) // 1e9 # type: ignore

    return ut

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline long[:] years(cnp.ndarray dt) noexcept:
    cdef long[:] Y = dt.astype("datetime64[Y]").astype(np.int64) + 1970 # type: ignore
    return Y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline long[:] months(cnp.ndarray dt) noexcept:
    cdef long[:] M = dt.astype("datetime64[M]").astype(np.int64) % 12 + 1 # type: ignore
    return M

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double julian_day(double unixtime) noexcept nogil: # type: ignore
    return unixtime * 1.0 / 86400 + 2440587.5

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double julian_ephemeris_day(double jd, double delta_t) noexcept nogil: # type: ignore
    return jd + delta_t * 1.0 / 86400

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double julian_century(double jd) noexcept nogil: # type: ignore
    return (jd - 2451545) * 1.0 / 36525

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double julian_ephemeris_century(double jde) noexcept nogil: # type: ignore
    return (jde - 2451545) * 1.0 / 36525

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef inline double  julian_ephemeris_millennium(double jce) noexcept nogil: # type: ignore
    return jce * 1.0 / 10

cdef double apparent_sidereal_time_at_greenwich(
    double jd, double jc, double E, double delta_psi) noexcept nogil # type: ignore


cdef double pe4dt(
    long year, long month, bint apply_corection) noexcept nogil # type: ignore


cdef double true_obliquity_of_the_ecliptic(
    double jme, double delta_eps) noexcept nogil  # type: ignore

cdef (double, double) geocentric_right_ascension_and_declination(          # 3.9.
    double apparent_lon, double geocentric_lat, double true_ecliptic_obliquity
) noexcept nogil # type: ignore
