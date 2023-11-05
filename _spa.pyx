# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# pyright: reportGeneralTypeIssues=false, reportMissingImports=false
cimport cython
from cython.parallel cimport prange

from libc.math cimport cos

cimport numpy as cnp
from numpy cimport float32_t as f32, float64_t as f64, ndarray as NDArray
import numpy as np

cnp.import_array()


# =============================================================================
# Julian Day Conversion
# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _julian_day(NDArray[f64, ndim=1] ut):
    cdef NDArray[f64, ndim=1] jd
    jd = ut * 1.0 / 86400 + 2440587.5
    return jd

def julian_day(NDArray[f64, ndim=1] ut):
    cdef NDArray[f64, ndim=1] jd
    jd = _julian_day(ut)
    return jd

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _julian_century(NDArray[f64, ndim=1] jd):
    cdef NDArray[f64, ndim=1] jc
    jc = (jd - 2451545) * 1.0 / 36525
    return jc

def julian_century(NDArray[f64, ndim=1] ut):
    cdef NDArray[f64, ndim=1] jd, jc
    jd = _julian_day(ut)
    jc = _julian_century(jd)
    return jc

# - ephemeris
@cython.boundscheck(False)
@cython.wraparound(False)
cdef _julian_ephemeris_day(NDArray[f64, ndim=1] jd, delta_t):
    cdef NDArray[f64, ndim=1] jde
    jde = jd + delta_t * 1.0 / 86400
    return jde

def julian_ephemeris_day(NDArray[f64, ndim=1] ut, delta_t):
    cdef NDArray[f64, ndim=1] jd, jde
    jd = _julian_day(ut)
    jde = _julian_ephemeris_day(jd, delta_t)
    return jde

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _julian_ephemeris_century(NDArray[f64, ndim=1] jde):
    cdef NDArray[f64, ndim=1] jce
    jce = (jde - 2451545) * 1.0 / 36525
    return jce

def julian_ephemeris_century(NDArray[f64, ndim=1] ut, delta_t):
    cdef NDArray[f64, ndim=1] jd, jde, jce
    jd = _julian_day(ut)
    jde = _julian_ephemeris_day(jd, delta_t)
    jce = _julian_ephemeris_century(jde)
    return jce

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _julian_ephemeris_millennium(NDArray[f64, ndim=1] jce):
    cdef NDArray[f64, ndim=1] jme
    jme = jce * 1.0 / 10
    return jme

def julian_ephemeris_millennium(NDArray[f64, ndim=1] ut,  delta_t):
    cdef NDArray[f64, ndim=1] jd, jde, jce, jme
    jd = _julian_day(ut)
    jde = _julian_ephemeris_day(jd, delta_t)
    jce = _julian_ephemeris_century(jde)
    jme = _julian_ephemeris_millennium(jce)
    return jme

def unix_time(x):
    cdef NDArray[f64, ndim=1] ut
    ut = np.asanyarray(x, dtype="datetime64[ns]").ravel().astype(np.float64) // 1e9
    return ut

# =============================================================================
#  Table A4.2. Earth Periodic Terms
# =============================================================================
L0 = np.array(
    [
        [175347046.0, 0.0, 0.0],
        [3341656.0, 4.6692568, 6283.07585],
        [34894.0, 4.6261, 12566.1517],
        [3497.0, 2.7441, 5753.3849],
        [3418.0, 2.8289, 3.5231],
        [3136.0, 3.6277, 77713.7715],
        [2676.0, 4.4181, 7860.4194],
        [2343.0, 6.1352, 3930.2097],
        [1324.0, 0.7425, 11506.7698],
        [1273.0, 2.0371, 529.691],
        [1199.0, 1.1096, 1577.3435],
        [990.0, 5.233, 5884.927],
        [902.0, 2.045, 26.298],
        [857.0, 3.508, 398.149],
        [780.0, 1.179, 5223.694],
        [753.0, 2.533, 5507.553],
        [505.0, 4.583, 18849.228],
        [492.0, 4.205, 775.523],
        [357.0, 2.92, 0.067],
        [317.0, 5.849, 11790.629],
        [284.0, 1.899, 796.298],
        [271.0, 0.315, 10977.079],
        [243.0, 0.345, 5486.778],
        [206.0, 4.806, 2544.314],
        [205.0, 1.869, 5573.143],
        [202.0, 2.458, 6069.777],
        [156.0, 0.833, 213.299],
        [132.0, 3.411, 2942.463],
        [126.0, 1.083, 20.775],
        [115.0, 0.645, 0.98],
        [103.0, 0.636, 4694.003],
        [102.0, 0.976, 15720.839],
        [102.0, 4.267, 7.114],
        [99.0, 6.21, 2146.17],
        [98.0, 0.68, 155.42],
        [86.0, 5.98, 161000.69],
        [85.0, 1.3, 6275.96],
        [85.0, 3.67, 71430.7],
        [80.0, 1.81, 17260.15],
        [79.0, 3.04, 12036.46],
        [75.0, 1.76, 5088.63],
        [74.0, 3.5, 3154.69],
        [74.0, 4.68, 801.82],
        [70.0, 0.83, 9437.76],
        [62.0, 3.98, 8827.39],
        [61.0, 1.82, 7084.9],
        [57.0, 2.78, 6286.6],
        [56.0, 4.39, 14143.5],
        [56.0, 3.47, 6279.55],
        [52.0, 0.19, 12139.55],
        [52.0, 1.33, 1748.02],
        [51.0, 0.28, 5856.48],
        [49.0, 0.49, 1194.45],
        [41.0, 5.37, 8429.24],
        [41.0, 2.4, 19651.05],
        [39.0, 6.17, 10447.39],
        [37.0, 6.04, 10213.29],
        [37.0, 2.57, 1059.38],
        [36.0, 1.71, 2352.87],
        [36.0, 1.78, 6812.77],
        [33.0, 0.59, 17789.85],
        [30.0, 0.44, 83996.85],
        [30.0, 2.74, 1349.87],
        [25.0, 3.16, 4690.48],
    ],
    dtype=np.float64,
)
L1 = np.array(
    [
        [628331966747.0, 0.0, 0.0],
        [206059.0, 2.678235, 6283.07585],
        [4303.0, 2.6351, 12566.1517],
        [425.0, 1.59, 3.523],
        [119.0, 5.796, 26.298],
        [109.0, 2.966, 1577.344],
        [93.0, 2.59, 18849.23],
        [72.0, 1.14, 529.69],
        [68.0, 1.87, 398.15],
        [67.0, 4.41, 5507.55],
        [59.0, 2.89, 5223.69],
        [56.0, 2.17, 155.42],
        [45.0, 0.4, 796.3],
        [36.0, 0.47, 775.52],
        [29.0, 2.65, 7.11],
        [21.0, 5.34, 0.98],
        [19.0, 1.85, 5486.78],
        [19.0, 4.97, 213.3],
        [17.0, 2.99, 6275.96],
        [16.0, 0.03, 2544.31],
        [16.0, 1.43, 2146.17],
        [15.0, 1.21, 10977.08],
        [12.0, 2.83, 1748.02],
        [12.0, 3.26, 5088.63],
        [12.0, 5.27, 1194.45],
        [12.0, 2.08, 4694.0],
        [11.0, 0.77, 553.57],
        [10.0, 1.3, 6286.6],
        [10.0, 4.24, 1349.87],
        [9.0, 2.7, 242.73],
        [9.0, 5.64, 951.72],
        [8.0, 5.3, 2352.87],
        [6.0, 2.65, 9437.76],
        [6.0, 4.67, 4690.48],
    ],
    dtype=np.float64,
)
L2 = np.array(
    [
        [52919.0, 0.0, 0.0],
        [8720.0, 1.0721, 6283.0758],
        [309.0, 0.867, 12566.152],
        [27.0, 0.05, 3.52],
        [16.0, 5.19, 26.3],
        [16.0, 3.68, 155.42],
        [10.0, 0.76, 18849.23],
        [9.0, 2.06, 77713.77],
        [7.0, 0.83, 775.52],
        [5.0, 4.66, 1577.34],
        [4.0, 1.03, 7.11],
        [4.0, 3.44, 5573.14],
        [3.0, 5.14, 796.3],
        [3.0, 6.05, 5507.55],
        [3.0, 1.19, 242.73],
        [3.0, 6.12, 529.69],
        [3.0, 0.31, 398.15],
        [3.0, 2.28, 553.57],
        [2.0, 4.38, 5223.69],
        [2.0, 3.75, 0.98],
    ],
    dtype=np.float64,
)
L3 = np.array(
    [
        [289.0, 5.844, 6283.076],
        [35.0, 0.0, 0.0],
        [17.0, 5.49, 12566.15],
        [3.0, 5.2, 155.42],
        [1.0, 4.72, 3.52],
        [1.0, 5.3, 18849.23],
        [1.0, 5.97, 242.73],
    ]
)
L4 = np.array(
    [
        [114.0, 3.142, 0.0],
        [8.0, 4.13, 6283.08],
        [1.0, 3.84, 12566.15],
    ],
    dtype=np.float64,
)
L5 = np.array(
    [
        [1.0, 3.14, 0.0],
    ],
    dtype=np.float64,
)
# -
B0 = np.array(
    [
        [280.0, 3.199, 84334.662],
        [102.0, 5.422, 5507.553],
        [80.0, 3.88, 5223.69],
        [44.0, 3.7, 2352.87],
        [32.0, 4.0, 1577.34],
    ],
    dtype=np.float64,
)
B1 = np.array(
    [
        [9.0, 3.9, 5507.55],
        [6.0, 1.73, 5223.69],
    ],
    dtype=np.float64,
)
# - heliocentric radius coefficients
R0 = np.array(
    [
        [100013989.0, 0.0, 0.0],
        [1670700.0, 3.0984635, 6283.07585],
        [13956.0, 3.05525, 12566.1517],
        [3084.0, 5.1985, 77713.7715],
        [1628.0, 1.1739, 5753.3849],
        [1576.0, 2.8469, 7860.4194],
        [925.0, 5.453, 11506.77],
        [542.0, 4.564, 3930.21],
        [472.0, 3.661, 5884.927],
        [346.0, 0.964, 5507.553],
        [329.0, 5.9, 5223.694],
        [307.0, 0.299, 5573.143],
        [243.0, 4.273, 11790.629],
        [212.0, 5.847, 1577.344],
        [186.0, 5.022, 10977.079],
        [175.0, 3.012, 18849.228],
        [110.0, 5.055, 5486.778],
        [98.0, 0.89, 6069.78],
        [86.0, 5.69, 15720.84],
        [86.0, 1.27, 161000.69],
        [65.0, 0.27, 17260.15],
        [63.0, 0.92, 529.69],
        [57.0, 2.01, 83996.85],
        [56.0, 5.24, 71430.7],
        [49.0, 3.25, 2544.31],
        [47.0, 2.58, 775.52],
        [45.0, 5.54, 9437.76],
        [43.0, 6.01, 6275.96],
        [39.0, 5.36, 4694.0],
        [38.0, 2.39, 8827.39],
        [37.0, 0.83, 19651.05],
        [37.0, 4.9, 12139.55],
        [36.0, 1.67, 12036.46],
        [35.0, 1.84, 2942.46],
        [33.0, 0.24, 7084.9],
        [32.0, 0.18, 5088.63],
        [32.0, 1.78, 398.15],
        [28.0, 1.21, 6286.6],
        [28.0, 1.9, 6279.55],
        [26.0, 4.59, 10447.39],
    ],
    dtype=np.float64,
)
R1 = np.array(
    [
        [103019.0, 1.10749, 6283.07585],
        [1721.0, 1.0644, 12566.1517],
        [702.0, 3.142, 0.0],
        [32.0, 1.02, 18849.23],
        [31.0, 2.84, 5507.55],
        [25.0, 1.32, 5223.69],
        [18.0, 1.42, 1577.34],
        [10.0, 5.91, 10977.08],
        [9.0, 1.42, 6275.96],
        [9.0, 0.27, 5486.78],
    ],
    dtype=np.float64,
)
R2 = np.array(
    [
        [4359.0, 5.7846, 6283.0758],
        [124.0, 5.579, 12566.152],
        [12.0, 3.14, 0.0],
        [9.0, 3.63, 77713.77],
        [6.0, 1.87, 5573.14],
        [3.0, 5.47, 18849.23],
    ],
    dtype=np.float64,
)
R3 = np.array(
    [
        [145.0, 4.273, 6283.076],
        [7.0, 3.92, 12566.15],
    ],
    dtype=np.float64,
)
R4 = np.array(
    [
        [4.0, 2.56, 6283.08],
    ],
    dtype=np.float64,
)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _term_calc(double[:,:] terms, double[:] jme):
    """
    >>> cdef double[:] input_ = np.arange(100, dtype=np.float64)
    """
    cdef int i, j, n_terms, n_jme
    cdef double A, B, C, JME
    cdef NDArray[f64, ndim=1] out
    n_terms = len(terms)
    n_jme = len(jme)
    out = np.zeros_like(jme, dtype=np.float64)
    for i in prange(n_terms, nogil=True):
        for j in range(n_jme):
            A, B, C = terms[i, 0], terms[i, 1], terms[i, 2]
            JME = jme[j]
            out[j] += A * cos(B + C * JME)
            
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _earth_heliocentric_longitude_latitude_and_radius_vector(NDArray[f64, ndim=1] jme):
    """
    3.2.	 Calculate the Earth heliocentric longitude, latitude, and radius vector (L, B, # and R): 
    “Heliocentric” means that the Earth position is calculated with respect to the center of the sun. 
    3.2.1. For each row of Table A4.2, calculate the term L0i (in radians), 
    L0 =	 A *cos ( B + C * JME) , (9) i i i i 
    where, 
    - i is the ith row for the term L0 in Table A4.2. 
    - Ai , Bi , and Ci are the values in the ith row and A, B, and C columns in Table 
    A4.2, for the term L0 (in radians). 
    3.2.2. Calculate the term L0 (in radians), 
    n 
    L0 = ∑ L0i ,	 (10) 
    i = 0 
    where n is the number of rows for the term L0 in Table A4.2. 
    3.2.3. Calculate the terms L1, L2, L3, L4, and L5 by using Equations 9 and 10 and 
    changing the 0 to 1, 2, 3, 4, and 5, and by using their corresponding values in"""
    # cdef NDArray L, B, R, l0, l1, l2, l3, l4, l5, b0, b1, r0, r1, r2, r3, r4
    cdef double[:] jme_view = jme
    l0 = _term_calc(L0, jme_view)
    l1 = _term_calc(L1, jme_view)
    l2 = _term_calc(L2, jme_view)
    l3 = _term_calc(L3, jme_view)
    l4 = _term_calc(L4, jme_view)
    l5 = _term_calc(L5, jme_view)
    L = np.rad2deg(
        (l0 + l1 * jme + l2 * jme**2 + l3 * jme**3 + l4 * jme**4 + l5 * jme**5) / 1e8
    ) % 360

    b0 = _term_calc(B0, jme_view)
    b1 = _term_calc(B1, jme_view)
    B = np.rad2deg((b0 + b1 * jme) / 1e8)

    # - radius vector (R) in astronomical units (AU)
    r0 = _term_calc(R0, jme_view)
    r1 = _term_calc(R1, jme_view)
    r2 = _term_calc(R2, jme_view)
    r3 = _term_calc(R3, jme_view)
    r4 = _term_calc(R4, jme_view)

    R = (r0 + r1 * jme + r2 * jme**2 + r3 * jme**3 + r4 * jme**4) / 1e8
    
    return L, B, R

# =============================================================================
cdef _fast_spa(
    tuple[int, int, int, int] shape,
    NDArray[f64, ndim=1] unixtime, 
    NDArray[f64, ndim=2] lattitude, 
    NDArray[f64, ndim=2] longitude, 
    float elev, 
    pressure, 
    temp, 
    delta_t, 
    atmos_refract,
    sst,
    esd,
):
    jd = _julian_day(unixtime)
    jde = julian_ephemeris_day(jd, delta_t)
    jc = _julian_century(jd)
    jce = _julian_ephemeris_century(jde)
    jme = _julian_ephemeris_millennium(jce)
    # 
    L, B, R = _earth_heliocentric_longitude_latitude_and_radius_vector(jme)
    return L, B, R

def fast_spa(
    unixtime,
    lat,
    lon,
    elev, 
    pressure, 
    temp, 
    delta_t, 
    atmos_refract
):

    assert lat.shape  == lon.shape

    return _fast_spa(
        (5, len(unixtime),) + lon.shape,
        unixtime, 
        lat, 
        lon, 
        elev, 
        pressure, 
        temp, 
        delta_t, 
        atmos_refract, 
        False,
        False,
    )