# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

# pyright: reportGeneralTypeIssues=false, reportUnusedExpression=false, reportMissingImports=false

from cython.view cimport array as cvarray

from numpy cimport (
    NPY_TYPES,
    ndarray as NDArray,
    PyArray_Cast,
)
import numpy as np
cimport numpy as np
np.import_array()


# =============================================================================
# - math
# =============================================================================
cdef extern from "<math.h>" nogil:
    double sin(double x)
    double cos(double x)
    double tan(double x)
    double asin(double x)
    double acos(double x)
    double atan(double x)
    double atan2(double y, double x)
    const double pi "M_PI"  # as in Python's math module
    
cdef inline double radians(double deg) noexcept nogil: # type: ignore
    return deg * (pi / 180)

cdef inline double degrees(double rad) noexcept nogil: # type: ignore
    return (rad * 180) / pi

cdef inline double arctan(double x) noexcept nogil: # type: ignore
    return atan2(x, 1)

cdef inline double arcsin(double x) noexcept nogil: # type: ignore
    return asin(x)

cdef inline double arctan2(double y, double x) noexcept nogil: # type: ignore
    return atan2(y, x)

# =============================================================================
# - memoryview
# =============================================================================
cdef inline viewnd(tuple shape) noexcept:
    return cvarray(shape, itemsize=8, format='d')

cdef inline double[:] view1d(int a) noexcept:
    cdef  double[:] out = viewnd((a,))
    return out

cdef inline double[:, :] view2d(int a, int b) noexcept:
    cdef  double[:, :] out = viewnd((a, b))
    return out

cdef inline double[:, :, :] view3d(int a, int b, int c) noexcept:
    cdef  double[:, :, :] out = viewnd((a, b, c))
    return out

cdef inline double[:, :, :, :] view4d(int a, int b, int c, int d) noexcept:
    cdef  double[:, :, :, :] out = viewnd((a, b, c, d))
    return out

# =============================================================================
# - array
# =============================================================================
cdef inline NDArray cast_array(NDArray a, int n) noexcept:
    return PyArray_Cast(a, n)

cdef inline NDArray darray1d(object x, int size) noexcept:
    cdef NDArray out
    out = np.asfarray(x, dtype=np.float64)
    if out.ndim == 0:
        return out[np.newaxis]
    elif out.ndim > 1:
        out = out.ravel()
    assert out.size == size
    return out

cdef inline NDArray dtarray(object datetime_like) noexcept:
    """
    main entry point for datetime_like.
    need to add validation to the object so everthing else can be marked as
    `nogil`
    """
    cdef NDArray dt
    dt = np.asanyarray(datetime_like, dtype="datetime64[ns]") # type: ignore
    return dt


cdef inline long[:] years(NDArray dt) noexcept:
    cdef long[:] Y 
    Y = dt.astype("datetime64[Y]").astype(np.int64) + 1970
    return Y

cdef inline long[:] months(NDArray dt) noexcept:
    cdef long[:] M
    M = dt.astype("datetime64[M]").astype(np.int64) % 12 + 1
    return M

# =============================================================================
# - time
# =============================================================================
cdef inline double[:] unixtime(NDArray dt) noexcept:
    cdef double[:] ut
    ut = cast_array(dt, NPY_TYPES.NPY_DOUBLE) // 1e9
    return ut

# - scalar functions
cdef inline double julian_day(double unixtime) noexcept nogil: # type: ignore
    return unixtime * 1.0 / 86400 + 2440587.5

cdef inline double julian_ephemeris_day(double jd, double delta_t) noexcept nogil: # type: ignore
    return jd + delta_t * 1.0 / 86400

cdef inline double julian_century(double jd) noexcept nogil: # type: ignore
    return (jd - 2451545) * 1.0 / 36525

cdef inline double julian_ephemeris_century(double jde) noexcept nogil: # type: ignore
    return (jde - 2451545) * 1.0 / 36525

cdef inline double  julian_ephemeris_millennium(double jce) noexcept nogil: # type: ignore
    return jce * 1.0 / 10

cdef double apparent_sidereal_time_at_greenwich(
    double jd, double jc, double E, double delta_psi) noexcept nogil # type: ignore

cdef double pe4dt(
    long year, long month, bint apply_corection) noexcept nogil # type: ignore

cdef double true_obliquity_of_the_ecliptic(
    double jme, double delta_eps) noexcept nogil  # type: ignore

cdef (double, double) geocentric_right_ascension_and_declination(
    double apparent_lon, double geocentric_lat, double true_ecliptic_obliquity
) noexcept nogil # type: ignore
