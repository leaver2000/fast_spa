# cython: language_level=3

# pyright: reportGeneralTypeIssues=false
import cython
cimport cython
from cython.parallel cimport prange  # type: ignore

import numpy as np
cimport numpy as np

from ._lib cimport degrees, radians, sin, cos

np.import_array()

# =============================================================================
# - nutation
# =============================================================================
# HELIOCENTRIC LONGITUDE TERMS
cdef const double[:, :] L0 = np.array(
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
    dtype=np.float64
)
cdef const double[:, :] L1 = np.array(
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
    dtype=np.float64
)
cdef const double[:, :] L2 = np.array(
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
    dtype=np.float64
)
cdef const double[:, :] L3 = np.array(
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
cdef const double[:, :] L4 = np.array(
    [
        [114.0, 3.142, 0.0], 
        [8.0, 4.13, 6283.08], 
        [1.0, 3.84, 12566.15]
    ], 
    dtype=np.float64
)
cdef const double[:, :] L5 = np.array(
    [
        [1.0, 3.14, 0.0]
    ],
    dtype=np.float64
)

# HELIOCENTRIC LATITUDE TERMS
cdef const double[:, :] B0 = np.array(
    [
        [280.0, 3.199, 84334.662],
        [102.0, 5.422, 5507.553],
        [80.0, 3.88, 5223.69],
        [44.0, 3.7, 2352.87],
        [32.0, 4.0, 1577.34]
    ], 
    dtype=np.float64
)
cdef const double[:, :] B1 = np.array(
    [
        [9.0, 3.9, 5507.55], 
        [6.0, 1.73, 5223.69]
    ], 
    dtype=np.float64
)


# heliocentric radius terms
cdef const double[:, :] R0 = np.array(
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
    dtype=np.float64
)
cdef const double[:, :] R1 = np.array(
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
    dtype=np.float64
)
cdef const double[:, :] R2 = np.array(
    [
        [4359.0, 5.7846, 6283.0758],
        [124.0, 5.579, 12566.152],
        [12.0, 3.14, 0.0],
        [9.0, 3.63, 77713.77],
        [6.0, 1.87, 5573.14],
        [3.0, 5.47, 18849.23],
    ], dtype=np.float64
)
cdef const double[:, :] R3 = np.array(
    [
        [145.0, 4.273, 6283.076],
        [7.0, 3.92, 12566.15]
    ], 
    dtype=np.float64
)
cdef const double[:, :] R4 = np.array(
    [
        [4.0, 2.56, 6283.08]
    ],
    dtype=np.float64
)


# =============================================================================
# 3.2. Calculate the Earth heliocentric longitude, latitude, and radius vector
# (L, B, R):
# =============================================================================

# 3.2.1. For each row of Table A4.2, calculate the term L0i(in radians)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _heilo(
    double jme, const double[:, :] terms, int num_threads) noexcept nogil: # type: ignore
    cdef int i, n
    cdef double A, B, C, x

    n = len(terms)
    x = 0.0

    for i in prange(n, nogil=True, num_threads=num_threads):
        A = terms[i, 0]
        B = terms[i, 1]
        C = terms[i, 2]
        x += A * cos(B + C * jme) 

    return x


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double heliocentric_longitude(double jme, int num_threads) noexcept nogil: # type: ignore
    """
    Heliocentric means that the Earth position is calculated with respect to 
    the center of the sun.
    """
    cdef double L, l0, l1, l2, l3, l4, l5

    l0 = _heilo(jme, L0, num_threads=num_threads)
    l1 = _heilo(jme, L1, num_threads=num_threads)
    l2 = _heilo(jme, L2, num_threads=num_threads)
    l3 = _heilo(jme, L3, num_threads=num_threads)
    l4 = _heilo(jme, L4, num_threads=num_threads)
    l5 = _heilo(jme, L5, num_threads=num_threads)
    # 3.2.4.
    L = (
        l0 + l1 * jme + l2 * jme**2 + l3 * jme**3 + l4 * jme**4 + l5 * jme**5
    )  / 1e8

    return degrees(L) % 360


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double heliocentric_latitude(double jme, int num_threads) noexcept nogil: # type: ignore
    """
    Heliocentric means that the Earth position is calculated with respect to 
    the center of the sun.
    """
    cdef double B, b0, b1

    b0 = _heilo(jme, B0, num_threads=num_threads)
    b1 = _heilo(jme, B1, num_threads=num_threads)
    B = (b0 + b1 * jme) / 1e8

    return degrees(B)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double heliocentric_radius_vector(double jme, int num_threads)  noexcept nogil: # type: ignore
    """
    Heliocentric means that the Earth position is calculated with respect to 
    the center of the sun.
    """
    cdef double R, r0, r1, r2, r3, r4

    r0 = _heilo(jme, R0, num_threads=num_threads)
    r1 = _heilo(jme, R1, num_threads=num_threads)
    r2 = _heilo(jme, R2, num_threads=num_threads)
    r3 = _heilo(jme, R3, num_threads=num_threads)
    r4 = _heilo(jme, R4, num_threads=num_threads)
    R = (r0 + r1 * jme + r2 * jme**2 + r3 * jme**3 + r4 * jme**4) / 1e8

    return R
# =============================================================================
# 3.3. Calculate the geocentric longitude and latitude (Θ and β)
# =============================================================================
cdef const double[:, :] LONGITUDE_AND_OBLIQUITY_NUTATION_SIN_COEFFICIENTS = np.array(
    [   # Y0 Y1 Y2 Y3 Y4
        [0, 0, 0, 0, 1],
        [-2, 0, 0, 2, 2],
        [0, 0, 0, 2, 2],
        [0, 0, 0, 0, 2],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [-2, 1, 0, 2, 2],
        [0, 0, 0, 2, 1],
        [0, 0, 1, 2, 2],
        [-2, -1, 0, 2, 2],
        [-2, 0, 1, 0, 0],
        [-2, 0, 0, 2, 1],
        [0, 0, -1, 2, 2],
        [2, 0, 0, 0, 0],
        [0, 0, 1, 0, 1],
        [2, 0, -1, 2, 2],
        [0, 0, -1, 0, 1],
        [0, 0, 1, 2, 1],
        [-2, 0, 2, 0, 0],
        [0, 0, -2, 2, 1],
        [2, 0, 0, 2, 2],
        [0, 0, 2, 2, 2],
        [0, 0, 2, 0, 0],
        [-2, 0, 1, 2, 2],
        [0, 0, 0, 2, 0],
        [-2, 0, 0, 2, 0],
        [0, 0, -1, 2, 1],
        [0, 2, 0, 0, 0],
        [2, 0, -1, 0, 1],
        [-2, 2, 0, 2, 2],
        [0, 1, 0, 0, 1],
        [-2, 0, 1, 0, 1],
        [0, -1, 0, 0, 1],
        [0, 0, 2, -2, 0],
        [2, 0, -1, 2, 1],
        [2, 0, 1, 2, 2],
        [0, 1, 0, 2, 2],
        [-2, 1, 1, 0, 0],
        [0, -1, 0, 2, 2],
        [2, 0, 0, 2, 1],
        [2, 0, 1, 0, 0],
        [-2, 0, 2, 2, 2],
        [-2, 0, 1, 2, 1],
        [2, 0, -2, 0, 1],
        [2, 0, 0, 0, 1],
        [0, -1, 1, 0, 0],
        [-2, -1, 0, 2, 1],
        [-2, 0, 0, 0, 1],
        [0, 0, 2, 2, 1],
        [-2, 0, 2, 0, 1],
        [-2, 1, 0, 2, 1],
        [0, 0, 1, -2, 0],
        [-1, 0, 1, 0, 0],
        [-2, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 2, 0],
        [0, 0, -2, 2, 2],
        [-1, -1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, -1, 1, 2, 2],
        [2, -1, -1, 2, 2],
        [0, 0, 3, 2, 2],
        [2, -1, 0, 2, 2],
    ], 
    dtype=np.float64
)

cdef const double[:, :] LONGITUDE_AND_OBLIQUITY_NUTATION_COEFFICIENTS = np.array(
    [   # A B C D
        [-171996, -174.2, 92025, 8.9],
        [-13187, -1.6, 5736, -3.1],
        [-2274, -0.2, 977, -0.5],
        [2062, 0.2, -895, 0.5],
        [1426, -3.4, 54, -0.1],
        [712, 0.1, -7, 0],
        [-517, 1.2, 224, -0.6],
        [-386, -0.4, 200, 0],
        [-301, 0, 129, -0.1],
        [217, -0.5, -95, 0.3],
        [-158, 0, 0, 0],
        [129, 0.1, -70, 0],
        [123, 0, -53, 0],
        [63, 0, 0, 0],
        [63, 0.1, -33, 0],
        [-59, 0, 26, 0],
        [-58, -0.1, 32, 0],
        [-51, 0, 27, 0],
        [48, 0, 0, 0],
        [46, 0, -24, 0],
        [-38, 0, 16, 0],
        [-31, 0, 13, 0],
        [29, 0, 0, 0],
        [29, 0, -12, 0],
        [26, 0, 0, 0],
        [-22, 0, 0, 0],
        [21, 0, -10, 0],
        [17, -0.1, 0, 0],
        [16, 0, -8, 0],
        [-16, 0.1, 7, 0],
        [-15, 0, 9, 0],
        [-13, 0, 7, 0],
        [-12, 0, 6, 0],
        [11, 0, 0, 0],
        [-10, 0, 5, 0],
        [-8, 0, 3, 0],
        [7, 0, -3, 0],
        [-7, 0, 0, 0],
        [-7, 0, 3, 0],
        [-7, 0, 3, 0],
        [6, 0, 0, 0],
        [6, 0, -3, 0],
        [6, 0, -3, 0],
        [-6, 0, 3, 0],
        [-6, 0, 3, 0],
        [5, 0, 0, 0],
        [-5, 0, 3, 0],
        [-5, 0, 3, 0],
        [-5, 0, 3, 0],
        [4, 0, 0, 0],
        [4, 0, 0, 0],
        [4, 0, 0, 0],
        [-4, 0, 0, 0],
        [-4, 0, 0, 0],
        [-4, 0, 0, 0],
        [3, 0, 0, 0],
        [-3, 0, 0, 0],
        [-3, 0, 0, 0],
        [-3, 0, 0, 0],
        [-3, 0, 0, 0],
        [-3, 0, 0, 0],
        [-3, 0, 0, 0],
        [-3, 0, 0, 0],
    ], 
    dtype=np.float64
)
assert len(LONGITUDE_AND_OBLIQUITY_NUTATION_SIN_COEFFICIENTS) == len(LONGITUDE_AND_OBLIQUITY_NUTATION_COEFFICIENTS)
cdef int NUM_LONGITUDE_AND_OBLIQUITY_NUTATION_COEFFICIENTS = len(LONGITUDE_AND_OBLIQUITY_NUTATION_SIN_COEFFICIENTS)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef (double, double) nutation_in_longitude_and_obliquity(
    double jce, int num_threads
) noexcept nogil: # type: ignore
    cdef int i
    cdef double A, B, C, D, X0, X1, X2, X3, X4, Y0, Y1, Y2, Y3, Y4, rads, delta_psi, delta_eps
    
    delta_psi = 0.0
    delta_eps = 0.0

    for i in prange(
        NUM_LONGITUDE_AND_OBLIQUITY_NUTATION_COEFFICIENTS, 
        nogil=True, 
        num_threads=num_threads
    ):
        Y0 = LONGITUDE_AND_OBLIQUITY_NUTATION_SIN_COEFFICIENTS[i, 0]
        Y1 = LONGITUDE_AND_OBLIQUITY_NUTATION_SIN_COEFFICIENTS[i, 1]
        Y2 = LONGITUDE_AND_OBLIQUITY_NUTATION_SIN_COEFFICIENTS[i, 2]
        Y3 = LONGITUDE_AND_OBLIQUITY_NUTATION_SIN_COEFFICIENTS[i, 3]
        Y4 = LONGITUDE_AND_OBLIQUITY_NUTATION_SIN_COEFFICIENTS[i, 4]

        A = LONGITUDE_AND_OBLIQUITY_NUTATION_COEFFICIENTS[i, 0]
        B = LONGITUDE_AND_OBLIQUITY_NUTATION_COEFFICIENTS[i, 1]
        C = LONGITUDE_AND_OBLIQUITY_NUTATION_COEFFICIENTS[i, 2]
        D = LONGITUDE_AND_OBLIQUITY_NUTATION_COEFFICIENTS[i, 3]

        # - in degrees
        # 3.4.1. Calculate the mean elongation of the moon from the sun
        X0 = 297.85036 + 445_267.111480 * jce - 0.0019142 * jce**2 + jce**3 / 189474

        # 3.4.2. Calculate the mean anomaly of the sun (Earth)
        X1 = 357.52772 + 35999.050340 * jce - 0.0001603 * jce**2 - jce**3 / 3e5

        # 3.4.3. Calculate the mean anomaly of the moon
        X2 = 134.96298 + 477_198.867398 * jce + 0.0086972 * jce**2 + jce**3 / 56250

        # 3.4.4. Calculate the moon’s argument of latitude
        X3 = (
            93.27191 + 483_202.017538 * jce - 0.0036825 * jce**2 + jce**3 / 327270
        )

        # 3.4.5. Calculate the longitude of the ascending node of the moon’s 
        # mean orbit on the ecliptic, measured from the mean equinox of the date
        X4 = (
            125.04452 - 1934.136261 * jce + 0.0020708 * jce**2 + jce**3 / 45e4
        )

        # - in radians
        rads = radians(Y0 * X0 + Y1 * X1 + Y2 * X2 + Y3 * X3 + Y4 * X4)

        # 3.4.7. Calculate the nutation in longitude
        delta_psi += (
            (A + B * jce) * sin(rads) * 1.0 / 36e6
        ) # ∆ψ = (ai + bi * jce ) *sin( ∑ X j *Yi, j ) 

        # 3.4.8. Calculate the nutation in obliquity
        delta_eps += (
            (C + D * jce) * cos(rads) * 1.0 / 36e6
        ) # ∆ε = (ci + di * jce ) *cos( ∑ X j *Yi, j )  

    return delta_psi, delta_eps

