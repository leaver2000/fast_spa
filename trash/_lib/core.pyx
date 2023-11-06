# pyright: reportGeneralTypeIssues=false, reportMissingImports=false

import numpy as np
cimport numpy as cnp

cnp.import_array()
cdef extern from "tabs.h":
    double L0[64][3] # type: ignore





cdef _get_tab():
    cdef double[:, ::1] l0_view = L0
    cdef cnp.ndarray[cnp.float64_t, ndim=2] l0 = np.asarray(l0_view)
    return l0


def get_tab():
    return _get_tab()