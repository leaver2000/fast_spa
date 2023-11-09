# 3.2. Calculate the Earth heliocentric longitude, latitude, and radius vector (L, B, and R)
cdef double heliocentric_longitude(double jme, int num_threads) noexcept nogil # type: ignore
cdef double heliocentric_latitude(double jme, int num_threads) noexcept nogil # type: ignore
cdef double heliocentric_radius_vector(double jme, int num_threads)  noexcept nogil # type: ignore

cdef (double, double) nutation_in_longitude_and_obliquity(double jce, int num_threads) noexcept nogil # type: ignore