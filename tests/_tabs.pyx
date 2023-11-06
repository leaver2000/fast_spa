# pyright: reportGeneralTypeIssues=false, reportMissingImports=false
# cimport numpy as cnp
# cimport cython
# import numpy as np

# import numpy as np
# =====================================================================================================================
#  Table A4.2. Earth Periodic Terms
# =====================================================================================================================
cdef double[:, :] L0
# HELIOCENTRIC LONGITUDE TERMS
cdef inline double[:,:] heliocentric_longitude_terms():
    return L0