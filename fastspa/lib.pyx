# pyright: reportGeneralTypeIssues=false, reportMissingImports=false

cimport cython

# cimport tab
cdef extern from "fastspa/tab.h":
    double[:,:] get_tab(int n)
cdef void f():
    print(get_tab(1))

def yerp():
    f()