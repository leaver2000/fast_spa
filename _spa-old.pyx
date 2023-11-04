# pyright: reportGeneralTypeIssues=false
"""https://www.nrel.gov/docs/fy08osti/34302.pdf"""
import numpy as np
cimport numpy as cnp
# from cpython cimport array
# from cython.view cimport array as cvarray
import array
from  numpy  cimport ndarray as NDArray
# from numpy cimport float32_t as f32, float64_t as f64, ndarray as NDArray
cnp.import_array()
ctypedef cnp.float64_t F64
# ctypedef class _spa.Array[object NDArray[F64, ndim=3]]:
#     pass
# ctypedef fused FloatTypes:
#     double
#     float

DTYPE = np.float64
# Table A4.2. Earth Periodic Terms 
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
    dtype=DTYPE
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
    dtype=DTYPE
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
    dtype=DTYPE
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
    [[114.0, 3.142, 0.0], [8.0, 4.13, 6283.08], [1.0, 3.84, 12566.15]], 
    dtype=DTYPE
    )
L5 = np.array([[1.0, 3.14, 0.0]])

# heliocentric latitude coefficients
B0 = np.array(
    [
        [280.0, 3.199, 84334.662],
        [102.0, 5.422, 5507.553],
        [80.0, 3.88, 5223.69],
        [44.0, 3.7, 2352.87],
        [32.0, 4.0, 1577.34],
    ], 
    dtype=DTYPE
)
B1 = np.array([[9.0, 3.9, 5507.55], [6.0, 1.73, 5223.69]], 
    dtype=DTYPE)


# heliocentric radius coefficients
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
    dtype=DTYPE
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
    dtype=DTYPE
)
R2 = np.array(
    [
        [4359.0, 5.7846, 6283.0758],
        [124.0, 5.579, 12566.152],
        [12.0, 3.14, 0.0],
        [9.0, 3.63, 77713.77],
        [6.0, 1.87, 5573.14],
        [3.0, 5.47, 18849.23],
    ]
)
R3 = np.array([[145.0, 4.273, 6283.076], [7.0, 3.92, 12566.15]], 
    dtype=DTYPE)
R4 = np.array([[4.0, 2.56, 6283.08]], 
    dtype=DTYPE)


# longitude and obliquity nutation coefficients
NUTATION_ABCD_ARRAY = np.array(
    [
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
    dtype=DTYPE
)

NUTATION_YTERM_ARRAY = np.array(
    [
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
    dtype=DTYPE
)



cdef _sum_mult_cos_add_mult(NDArray[F64, ndim=2] x, NDArray[F64, ndim=3] jme):
    cdef int i, size_x, n_time
    cdef NDArray[F64, ndim=3] out
    size_x = x.shape[0]
    n_time = jme.shape[0]
    out = np.zeros((n_time, 1 ,1), dtype=np.float64)
    for i in range(size_x):
        out += x[i, 0] * np.cos(x[i, 1] + x[i, 2] * jme)
    return out


cdef _heliocentric_longitude(cnp.ndarray[F64, ndim=3] jme):
    cdef NDArray[F64, ndim=3] l0, l1, l2, l3, l4, l5, l_rad, l
    l0 = _sum_mult_cos_add_mult(L0, jme)
    l1 = _sum_mult_cos_add_mult(L1, jme)
    l2 = _sum_mult_cos_add_mult(L2, jme)
    l3 = _sum_mult_cos_add_mult(L3, jme)
    l4 = _sum_mult_cos_add_mult(L4, jme)
    l5 = _sum_mult_cos_add_mult(L5, jme)
    l_rad = (l0 + l1 * jme + l2 * jme**2 + l3 * jme**3 + l4 * jme**4 + l5 * jme**5) / 10**8
    l = np.rad2deg(l_rad)
    return l % 360


cdef _heliocentric_latitude(NDArray[F64, ndim=3] jme):
    cdef NDArray[F64, ndim=3] b0, b1, b_rad, b
    b0 = _sum_mult_cos_add_mult(B0, jme)
    b1 = _sum_mult_cos_add_mult(B1, jme)
    b_rad = (b0 + b1 * jme) / 10**8
    b = np.rad2deg(b_rad)
    return b


cdef _heliocentric_radius_vector(NDArray[F64, ndim=3] jme):
    cdef NDArray[F64, ndim=3] r, r0, r1, r2, r3, r4
    r0 = _sum_mult_cos_add_mult(R0, jme)
    r1 = _sum_mult_cos_add_mult(R1, jme)
    r2 = _sum_mult_cos_add_mult(R2, jme)
    r3 = _sum_mult_cos_add_mult(R3, jme)
    r4 = _sum_mult_cos_add_mult(R4, jme)
    r = (r0 + r1 * jme + r2 * jme**2 + r3 * jme**3 + r4 * jme**4) / 10**8
    return r


cdef _geocentric_longitude(cnp.ndarray[F64, ndim=3] heliocentric_longitude):
    cdef NDArray[F64, ndim=3] theta
    theta = heliocentric_longitude + 180.0
    theta %= 360
    return theta


cdef _geocentric_latitude(NDArray[F64, ndim=3] heliocentric_latitude):
    cdef NDArray[F64, ndim=3] beta
    beta = -1.0 * heliocentric_latitude
    return beta

# 3.4. Calculate the nutation in longitude and obliquity ()R and )g): 
# 3.4.1. Calculate the mean elongation of the moon from the sun, X0 (in degrees), 
# X0 = 297,85036 . + 445267111480 . * JCE − (15) JCE 3 
# 0 0019142 . * JCE 2 + . 189474 
# 3.4.2. Calculate the mean anomaly of the sun (Earth), X1 (in degrees), 
# X1 = 357 52772 . + 35999 050340 . * JCE − 3 (16) 
# 2 JCE
# 0 0001603 . * JCE − . 300000 

# 3.4.3. Calculate the mean anomaly of the moon, X2 (in degrees), 
# X 2 = 134 96298 . + 477198 867398 . * JCE +

# 2 JCE 3 
# 0 0086972 . * JCE + . 56250 
# 3.4.4. Calculate the moon’s argument of latitude, X3 (in degrees), 
# X 3 = 9327191 . + 483202 017538 . * JCE −
# JCE 3 (18) 0 0036825 . * JCE 2 + . 327270 
# 3.4.5. Calculate the longitude of the ascending node of the moon’s mean orbit on the 
# ecliptic, measured from the mean equinox of the date, X4 (in degrees), 
# X 4 = 125 04452 . − 1934 136261 . * JCE +
# JCE 3 (19) 0 0020708 . * JCE 2 + . 450000 
# 3.4.6. For each row in Table A4.3, calculate the terms )Ri
#  and )gi
#  (in 0.0001of arc 
# seconds), 
# 4 
# ∆ψ i = (ai + bi * JCE ) *sin ( ∑ X j 
# *Yi, j ) , (20) 
# j = 0 
# 6 
 
# 4 
# ∆ ε i = (c + d * JCE ) *cos ( ∑ X *Y ) , (21) i i j i, j 
# j = 0 
# where, 
# - ai
#  , bi
#  , ci , and di are the values listed in the ith row and columns a, b, c, and d in 
# Table A4.3. 
# - X j
#  is the jth X calculated by using Equations 15 through 19. 
# - Yi, j is the value listed in ith row and jth Y column in Table A4.3. 
cdef _mean_elongation(NDArray[F64, ndim=3] jce):
    cdef NDArray[F64, ndim=3] x1
    x1 = 297.85036 + 445267.111480 * jce - 0.0019142 * jce**2 + jce**3 / 189474
    return x1


cdef _mean_anomaly_sun(NDArray[F64, ndim=3] jce):
    """3.4.3. Calculate the mean anomaly of the moon, X2 (in degrees), """
    cdef NDArray[F64, ndim=3] x1
    x1 = 357.52772 + 35999.050340 * jce - 0.0001603 * jce**2 - jce**3 / 300000
    return x1


cdef _mean_anomaly_moon(NDArray[F64, ndim=3] jce):
    cdef NDArray[F64, ndim=3] x2
    x2 = 134.96298 + 477198.867398 * jce + 0.0086972 * jce**2 + jce**3 / 56250
    return x2


cdef _moon_argument_latitude(NDArray[F64, ndim=3] jce):
    cdef NDArray[F64, ndim=3] x3
    x3 = 93.27191 + 483202.017538 * jce - 0.0036825 * jce**2 + jce**3 / 327270
    return x3


cdef _moon_ascending_longitude(NDArray[F64, ndim=3] jce):
    cdef NDArray[F64, ndim=3] x4
    x4 = 125.04452 - 1934.136261 * jce + 0.0020708 * jce**2 + jce**3 / 450000
    return x4
# Calculate the nutation in longitude and obliquity ()R and )g):
cdef _calculate_the_nutation_in_longitude_and_obliquity(NDArray[F64, ndim=3] jce):
    """3.4. Calculate the nutation in longitude and obliquity ()R and )g)"""
    cdef NDArray[F64, ndim=3] x0, x1, x2, x3, x4
    # 3.4.1. Calculate the mean elongation of the moon from the sun, X0 (in degrees)
    x0 = 297.85036 + 445_267.111480 * jce - 0.0019142 * jce**2 + jce**3 / 189_474
    # 3.4.2. Calculate the mean anomaly of the sun (Earth), X1 (in degrees)
    x1 = 357.52772 + 35999.050340 * jce - 0.0001603 * jce**2 - jce**3 / 3e5
    # 3.4.3. Calculate the mean anomaly of the moon, X2 (in degrees)
    x2 = 134.96298 + 477_198.867398 * jce + 0.0086972 * jce**2 + jce**3 / 56_250
    # 3.4.4. Calculate the moon’s argument of latitude, X3 (in degrees)
    x3 = 93.27191 + 483_202.017538 * jce - 0.0036825 * jce**2 + jce**3 / 327_270
    # 3.4.5. Calculate the longitude of the ascending node of the moon’s mean orbit on the
    # ecliptic, measured from the mean equinox of the date, X4 (in degrees)
    x4 = 125.04452 - 1_934.136261 * jce + 0.0020708 * jce**2 + jce**3 / 45e4


    return _longitude_obliquity_nutation(jce, x0, x1, x2, x3, x4)




cdef _longitude_obliquity_nutation(
    NDArray[F64, ndim=3] jce, 
    NDArray[F64, ndim=3] x0, 
    NDArray[F64, ndim=3] x1, 
    NDArray[F64, ndim=3] x2, 
    NDArray[F64, ndim=3] x3, 
    NDArray[F64, ndim=3] x4, 
):  
    cdef int row, n
    cdef float a, b, c, d, e
    cdef NDArray[F64, ndim=3] rads
    n = NUTATION_YTERM_ARRAY.shape[0]
    delta_psi = np.zeros_like(jce)
    delta_eps = np.zeros_like(jce)
    for row in range(n):
        a, b, c, d, e = NUTATION_YTERM_ARRAY[row]
        rads = np.radians(a * x0 + b * x1 + c * x2 + d * x3 + e * x4)
        a, b, c, d = NUTATION_ABCD_ARRAY[row]
        delta_psi += (a + b * jce) * np.sin(rads)
        delta_eps += (c + d * jce) * np.cos(rads)
    delta_psi *= 1.0 / 36_000_000
    delta_eps *= 1.0 / 36_000_000
    # seems like we ought to be able to return a tuple here instead
    # of resorting to `out`, but returning a UniTuple from this
    # function caused calculations elsewhere to give the wrong result.
    # very difficult to investigate since it did not occur when using
    # object mode.  issue was observed on numba 0.56.4
    return delta_psi, delta_eps

def true_ecliptic_obliquity(mean_ecliptic_obliquity, obliquity_nutation):
    e0 = mean_ecliptic_obliquity
    deleps = obliquity_nutation
    e = e0 * 1.0 / 3600 + deleps
    return e


def aberration_correction(earth_radius_vector):
    deltau = -20.4898 / (3600 * earth_radius_vector)
    return deltau


def apparent_sun_longitude(geocentric_longitude, longitude_nutation, aberration_correction):
    lamd = geocentric_longitude + longitude_nutation + aberration_correction
    return lamd


def mean_sidereal_time(julian_day, julian_century):
    v0 = (
        280.46061837
        + 360.98564736629 * (julian_day - 2451545)
        + 0.000387933 * julian_century**2
        - julian_century**3 / 38710000
    )
    return v0 % 360.0


def apparent_sidereal_time(mean_sidereal_time, longitude_nutation, true_ecliptic_obliquity):
    v = mean_sidereal_time + longitude_nutation * np.cos(np.radians(true_ecliptic_obliquity))
    return v


def geocentric_sun_right_ascension(apparent_sun_longitude, true_ecliptic_obliquity, geocentric_latitude):
    true_ecliptic_obliquity_rad = np.radians(true_ecliptic_obliquity)
    apparent_sun_longitude_rad = np.radians(apparent_sun_longitude)

    num = np.sin(apparent_sun_longitude_rad) * np.cos(true_ecliptic_obliquity_rad) - np.tan(
        np.radians(geocentric_latitude)
    ) * np.sin(true_ecliptic_obliquity_rad)
    alpha = np.degrees(np.arctan2(num, np.cos(apparent_sun_longitude_rad)))
    return alpha % 360


def geocentric_sun_declination(apparent_sun_longitude, true_ecliptic_obliquity, geocentric_latitude):
    geocentric_latitude_rad = np.radians(geocentric_latitude)
    true_ecliptic_obliquity_rad = np.radians(true_ecliptic_obliquity)

    delta = np.degrees(
        np.arcsin(
            np.sin(geocentric_latitude_rad) * np.cos(true_ecliptic_obliquity_rad)
            + np.cos(geocentric_latitude_rad)
            * np.sin(true_ecliptic_obliquity_rad)
            * np.sin(np.radians(apparent_sun_longitude))
        )
    )
    return delta


cdef _local_hour_angle(
    NDArray[F64, ndim=3] apparent_sidereal_time,
    NDArray[F64, ndim=3] observer_longitude,
    NDArray[F64, ndim=3] sun_right_ascension
):
    cdef NDArray[F64, ndim=3] x
    x = (apparent_sidereal_time + observer_longitude - sun_right_ascension) % 360.0
    return x
    


def equatorial_horizontal_parallax(earth_radius_vector):
    xi = 8.794 / (3600 * earth_radius_vector)
    return xi



cdef _termination_point(
    NDArray[F64, ndim=3] observer_latitude,
    float observer_elevation,
):
    # cdef NDArray[F64, ndim=3] u, x, y
    u = np.arctan(0.99664719 * np.tan(np.radians(observer_latitude)))
    x = np.cos(u) + observer_elevation / 6378140 * np.cos(np.radians(observer_latitude))
    y = 0.99664719 * np.sin(u) + observer_elevation / 6378140 * np.sin(np.radians(observer_latitude))
    return x, y


cdef _parallax_sun_right_ascension(
    NDArray[F64, ndim=3] xterm, 
    NDArray[F64, ndim=3] equatorial_horizontal_parallax, 
    NDArray[F64, ndim=3] local_hour_angle,
    NDArray[F64, ndim=3] geocentric_sun_declination,
):
    cdef NDArray[F64, ndim=3] equatorial_horizontal_parallax_rad, local_hour_angle_rad, num, denom, delta_alpha
    
    equatorial_horizontal_parallax_rad = np.radians(equatorial_horizontal_parallax)
    local_hour_angle_rad = np.radians(local_hour_angle)
    num = -xterm * np.sin(equatorial_horizontal_parallax_rad) * np.sin(local_hour_angle_rad)
    denom = np.cos(np.radians(geocentric_sun_declination)) - xterm * np.sin(equatorial_horizontal_parallax_rad) * np.cos(local_hour_angle_rad)
    delta_alpha = np.degrees(np.arctan2(num, denom))
    return delta_alpha


def topocentric_sun_right_ascension(geocentric_sun_right_ascension, parallax_sun_right_ascension):
    alpha_prime = geocentric_sun_right_ascension + parallax_sun_right_ascension
    return alpha_prime


cdef _topocentric_sun_declination(
    NDArray[F64, ndim=3] geocentric_sun_declination,
    NDArray[F64, ndim=3] xterm,
    NDArray[F64, ndim=3] yterm,
    NDArray[F64, ndim=3] equatorial_horizontal_parallax,
    NDArray[F64, ndim=3] parallax_sun_right_ascension,
    NDArray[F64, ndim=3] local_hour_angle,
):
    cdef NDArray[F64, ndim=3] geocentric_sun_declination_rad, equatorial_horizontal_parallax_rad, num, denom, delta
    geocentric_sun_declination_rad = np.radians(geocentric_sun_declination)
    equatorial_horizontal_parallax_rad = np.radians(equatorial_horizontal_parallax)

    num = (np.sin(geocentric_sun_declination_rad) - yterm * np.sin(equatorial_horizontal_parallax_rad)) * np.cos(
        np.radians(parallax_sun_right_ascension)
    )
    denom = np.cos(geocentric_sun_declination_rad) - xterm * np.sin(equatorial_horizontal_parallax_rad) * np.cos(
        np.radians(local_hour_angle)
    )
    delta = np.degrees(np.arctan2(num, denom))
    return delta


cdef _topocentric_local_hour_angle(
    NDArray[F64, ndim=3] local_hour_angle, 
    NDArray[F64, ndim=3] parallax_sun_right_ascension
):
    cdef NDArray[F64, ndim=3] H_prime
    H_prime = local_hour_angle - parallax_sun_right_ascension
    return H_prime


cdef _topocentric_elevation_angle_without_atmosphere(
    NDArray[F64, ndim=3] observer_latitude,
    NDArray[F64, ndim=3] topocentric_sun_declination,
    NDArray[F64, ndim=3] topocentric_local_hour_angle
):
    cdef NDArray[F64, ndim=3] observer_latitude_rad 
    cdef NDArray[F64, ndim=3] topocentric_sun_declination_rad, e0
    observer_latitude_rad = np.radians(observer_latitude)
    topocentric_sun_declination_rad = np.radians(topocentric_sun_declination)
    e0 = np.degrees(
        np.arcsin(
            np.sin(observer_latitude_rad) * np.sin(topocentric_sun_declination_rad)
            + np.cos(observer_latitude_rad)
            * np.cos(topocentric_sun_declination_rad)
            * np.cos(np.radians(topocentric_local_hour_angle))
        )
    )
    return e0


cdef _atmospheric_refraction_correction(
    NDArray[F64, ndim=3] elevation_angle, 
    float local_pressure, 
    float local_temp, 
    float atmos_refract
):
    cdef NDArray[F64, ndim=3] delta_e
    
    delta_e = (
        (local_pressure / 1010.0)
        * (283.0 / (273 + local_temp))
        * 1.02
        / (
            60
            * np.tan(
                np.radians(
                    elevation_angle
                    + 10.3 / (elevation_angle + 5.11)
                )
            )
        )
    ) * (elevation_angle >= -1.0 * (0.26667 + atmos_refract))
    return delta_e
# =============================================================================
# - topocentric
# =============================================================================
cdef _topocentric_elevation_angle(
    NDArray[F64, ndim=3] topocentric_elevation_angle_without_atmosphere, 
    NDArray[F64, ndim=3] atmospheric_refraction_correction,
):
    return topocentric_elevation_angle_without_atmosphere + atmospheric_refraction_correction


cdef _topocentric_zenith_angle(NDArray[F64, ndim=3] e0):
    return  90 - e0

cdef _topocentric_astronomers_azimuth(
    NDArray[F64, ndim=3] topocentric_local_hour_angle, 
    NDArray[F64, ndim=3] topocentric_sun_declination, 
    NDArray[F64, ndim=3] observer_latitude,
):
    cdef NDArray[F64, ndim=3] topocentric_local_hour_angle_rad, observer_latitude_rad, num, denom, gamma
    topocentric_local_hour_angle_rad = np.radians(topocentric_local_hour_angle)
    observer_latitude_rad = np.radians(observer_latitude)
    num = np.sin(topocentric_local_hour_angle_rad)
    denom = np.cos(topocentric_local_hour_angle_rad) * np.sin(observer_latitude_rad) - np.tan(
        np.radians(topocentric_sun_declination)
    ) * np.cos(observer_latitude_rad)
    gamma = np.degrees(np.arctan2(num, denom))
    return gamma % 360

cdef _topocentric_azimuth_angle(
    NDArray[F64, ndim=3] topocentric_astronomers_azimuth
):
    cdef NDArray[F64, ndim=3] phi
    phi = topocentric_astronomers_azimuth + 180
    return phi % 360


# =============================================================================
# - julian
# =============================================================================
cdef _julian_day(
    NDArray[F64, ndim=3] unixtime
):
    cdef NDArray[F64, ndim=3] jd
    jd = unixtime * 1.0 / 86400 + 2440587.5
    return jd


def julian_ephemeris_day(julian_day, delta_t):
    cdef NDArray[F64, ndim=3] jde
    jde = julian_day + delta_t * 1.0 / 86400
    return jde


cdef _julian_century(NDArray[F64, ndim=3] jd):
    cdef NDArray[F64, ndim=3] jc
    jc = (jd - 2451545) * 1.0 / 36525
    return jc


cdef _julian_ephemeris_century(NDArray[F64, ndim=3] jde):
    jce = (jde - 2451545) * 1.0 / 36525
    return jce


cdef _julian_ephemeris_millennium(NDArray[F64, ndim=3] jce):
    jme = jce * 1.0 / 10
    return jme

# =============================================================================


# def longitude_obliquity_nutation(
#     julian_ephemeris_century, 
#     x0, 
#     x1, 
#     x2, 
#     x3, 
#     x4, 
#     # out
# ):
#     delta_psi_sum = 0.0
#     delta_eps_sum = 0.0
#     for row in range(NUTATION_YTERM_ARRAY.shape[0]):
#         a = NUTATION_ABCD_ARRAY[row, 0]
#         b = NUTATION_ABCD_ARRAY[row, 1]
#         c = NUTATION_ABCD_ARRAY[row, 2]
#         d = NUTATION_ABCD_ARRAY[row, 3]
#         arg = np.radians(
#             NUTATION_YTERM_ARRAY[row, 0] * x0
#             + NUTATION_YTERM_ARRAY[row, 1] * x1
#             + NUTATION_YTERM_ARRAY[row, 2] * x2
#             + NUTATION_YTERM_ARRAY[row, 3] * x3
#             + NUTATION_YTERM_ARRAY[row, 4] * x4
#         )
#         delta_psi_sum += (a + b * julian_ephemeris_century) * np.sin(arg)
#         delta_eps_sum += (c + d * julian_ephemeris_century) * np.cos(arg)
#     delta_psi = delta_psi_sum * 1.0 / 36000000
#     delta_eps = delta_eps_sum * 1.0 / 36000000
#     # seems like we ought to be able to return a tuple here instead
#     # of resorting to `out`, but returning a UniTuple from this
#     # function caused calculations elsewhere to give the wrong result.
#     # very difficult to investigate since it did not occur when using
#     # object mode.  issue was observed on numba 0.56.4
#     return delta_psi, delta_eps


cdef _mean_ecliptic_obliquity(
    NDArray[F64, ndim=3] julian_ephemeris_millennium
):
    cdef NDArray[F64, ndim=3] U, e0
    U = 1.0 * julian_ephemeris_millennium / 10
    e0 = (
        84381.448
        - 4680.93 * U
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
    return e0

cdef _fast_spa(
    tuple[int, int, int, int] shape,
    NDArray[F64, ndim=1] unixtime, 
    NDArray[F64, ndim=2] lattitude, 
    NDArray[F64, ndim=2] longitude, 
    float elev, 
    pressure, 
    temp, 
    delta_t, 
    atmos_refract,
    sst,
    esd,
):
    cdef NDArray[F64, ndim=3] time, lat, lon, jd, jde, jc, jce, jme
    # cdef NDArray[F64, ndim=4] result = np.zeros(shape, dtype=DTYPE)
    time, lat, lon = unixtime[:, None, None], lattitude[None, ...], longitude[None, ...]

    jd = _julian_day(time)
    jde = julian_ephemeris_day(jd, delta_t)
    jc = _julian_century(jd)
    jce = _julian_ephemeris_century(jde)
    jme = _julian_ephemeris_millennium(jce)
    R = _heliocentric_radius_vector(jme)
    if esd:
        return (R,)
    L = _heliocentric_longitude(jme)
    B = _heliocentric_latitude(jme)
    Theta = _geocentric_longitude(L)
    beta = _geocentric_latitude(B)
    x0 = _mean_elongation(jce)
    x1 = _mean_anomaly_sun(jce)
    x2 = _mean_anomaly_moon(jce)
    x3 = _moon_argument_latitude(jce)
    x4 = _moon_ascending_longitude(jce)
    
    delta_psi, delta_epsilon = _longitude_obliquity_nutation(jce, x0, x1, x2, x3, x4)
    
    
    epsilon0 = _mean_ecliptic_obliquity(jme)
    epsilon = true_ecliptic_obliquity(epsilon0, delta_epsilon)
    delta_tau = aberration_correction(R)
    lamd = apparent_sun_longitude(Theta, delta_psi, delta_tau)
    v0 = mean_sidereal_time(jd, jc)
    v = apparent_sidereal_time(v0, delta_psi, epsilon)
    alpha = geocentric_sun_right_ascension(lamd, epsilon, beta)

    
    delta = geocentric_sun_declination(lamd, epsilon, beta)
    
    
    H = _local_hour_angle(v, lon, alpha)
    xi = equatorial_horizontal_parallax(R)
    
    if sst:
        return v, alpha, delta

    x, y = _termination_point(lat, elev)
    
    delta_alpha = _parallax_sun_right_ascension(x, xi, H, delta)
    delta_prime = _topocentric_sun_declination(delta, x, y, xi, delta_alpha, H)
    H_prime = _topocentric_local_hour_angle(H, delta_alpha)
    e0 = _topocentric_elevation_angle_without_atmosphere(lat, delta_prime, H_prime)
    e = _topocentric_elevation_angle(
        e0, _atmospheric_refraction_correction(e0, pressure, temp,  atmos_refract)
    )
    theta = _topocentric_zenith_angle(e)
    theta0 = _topocentric_zenith_angle(e0)
    phi = _topocentric_azimuth_angle(
        _topocentric_astronomers_azimuth(H_prime, delta_prime, lat)
    )
    return np.stack([theta, theta0, e, e0, phi])


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