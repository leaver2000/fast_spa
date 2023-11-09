cdef double longitude(double jme, int num_threads) noexcept nogil # type: ignore
cdef double latitude(double jme, int num_threads) noexcept nogil # type: ignore
cdef double radius_vector(double jme, int num_threads)  noexcept nogil # type: ignore
cdef (double, double) nutation_in_longitude_and_obliquity(
    double jce, int num_threads
) noexcept nogil # type: ignore