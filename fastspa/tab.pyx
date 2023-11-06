# pyright: reportGeneralTypeIssues=false, reportMissingImports=false
cimport cython
cimport numpy as cnp

import numpy as np
# cimport _term
# from _term cimport TermDish

cimport tab
from tab cimport spamdish
cnp.import_array()

cdef void prepare(spamdish *d):
    d.oz_of_spam = 42
    d.filler = otherstuff.oz_of_spam

def serve():
    cdef spamdish d
    prepare(&d)
    print(f'{d.oz_of_spam} oz spam, filler no. {d.filler}')