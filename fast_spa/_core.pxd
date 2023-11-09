# cython: language_level=3
cdef double[:,:] time_components(
    object datetime_like,
    bint apply_correction = ?, 
    int num_threads = ?
)