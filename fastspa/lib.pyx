# pyright: reportGeneralTypeIssues=false, reportMissingImports=false

import cython
cimport numpy as cnp
import numpy as np
from . cimport tab
# cimport _tabs
# cdef _get_tab():
#     return _tabs.L0

# cdef _get_tab():
#     return tab.prepare
#     ...

def get_tab():
    ...