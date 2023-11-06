
cimport cython
cimport cython

cdef extern from "fastspa/tab.h":
    double[:,:] get_tab(int n)