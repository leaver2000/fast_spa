# pyright: reportGeneralTypeIssues=false, reportMissingImports=false
cimport cython

import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef double[:,:] L0
L0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)



cdef double[:,:] get_tab(int n):
    return L0

