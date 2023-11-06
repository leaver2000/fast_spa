# pyright: reportGeneralTypeIssues=false, reportMissingImports=false
cimport cython
cimport numpy as cnp
cnp.import_array()
# =====================================================================================================================
#  Table A4.2. Earth Periodic Terms
# =====================================================================================================================
cdef double[:,:] L0
cdef double[:,:] get_tab(int n)