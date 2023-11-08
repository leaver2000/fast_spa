# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# pyright: reportGeneralTypeIssues=false, reportMissingImports=false
# import cython
cimport cython
from cython.view cimport array as cvarray
cimport numpy as np
ctypedef double f64
ctypedef float f32
ctypedef signed long long i64
ctypedef signed int i32
ctypedef unsigned long long u64
ctypedef unsigned int u32



cdef inline fview(tuple shape) noexcept:
    return cvarray(shape, itemsize=8, format='d')


cdef inline double[:] view1d(int a) noexcept:
    cdef  double[:] out = fview((a,))
    return out
    
cdef inline double[:, :] view2d(int a, int b) noexcept:
    cdef  double[:, :] out = fview((a, b))
    return out

cdef inline double[:, :, :] view3d(int a, int b, int c) noexcept:
    cdef  double[:, :, :] out = fview((a, b, c))
    return out

cdef inline np.ndarray cast_array(np.ndarray a, int n) noexcept:
    return np.PyArray_Cast(a, n) # type: ignore


# cdef inline f64[:, :] view2d(int a, int b):
#     return fview((a, b))


