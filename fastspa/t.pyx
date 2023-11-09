# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import numpy as np
cimport numpy as cnp
from numpy import sin, cos, tan, arctan, arctan2, arcsin, degrees, radians
from numpy.typing import ArrayLike
cnp.import_array()
cimport fastspa._core as fastspa


# cimport fastspa._lib as lib
cdef int TOPOCENTRIC_RIGHT_ASCENSION = 0
cdef int TOPOCENTRIC_DECLINATION = 1
cdef int APARENT_SIDEREAL_TIME = 2
cdef int EQUATOIRAL_HORIZONAL_PARALAX = 3


ctypedef cnp.float64_t DTYPE_t 
DTYPE = np.float64



cdef  topocentric_parallax_right_ascension_and_declination(
    double delta,   # δ geocentric sun declination
    cnp.ndarray[DTYPE_t, ndim=2] H,       # H local hour angle
    cnp.ndarray[DTYPE_t, ndim=2] E,       # E observer elevation
    cnp.ndarray[DTYPE_t, ndim=2] lat,     # observer latitude
    double xi,      # ξ equatorial horizontal parallax
) : # type: ignore
    cdef cnp.ndarray[DTYPE_t, ndim=2] u, x, y, Phi, delta_alpha, delta_p
    
    delta = radians(delta)          # δ
    xi = radians(xi)                # ξ
    Phi = radians(lat)              # ϕ

    # - 3.12.2. Calculate the term u (in radians)
    u = (
        arctan(0.99664719 * tan(Phi))
    ) # u = Acrtan(0.99664719 * tan ϕ)

    # - 3.12.3. Calculate the term x,
    x = (
        cos(u) + E / 6378140 * cos(Phi)
    ) # x = cosu + E / 6378140 * cos ϕ

    # - 3.12.4. Calculate the term y
    y = (
        0.99664719 * sin(u) + E / 6378140 * sin(Phi)
    ) # y = 0.99664719 * sin u + E / 6378140 * sin ϕ

    # - 3.12.5. Calculate the parallax in the sun right ascension (in degrees),
    delta_alpha = arctan2(
        -x * sin(xi) * sin(H),
        cos(delta) - x * sin(xi) * cos(H)
    ) # ∆α = Arctan2(-x *sin ξ *sin H / cosδ − x * sin ξ * cos H)

    delta_p = arcsin(
        sin(delta) - y * sin(xi) * cos(delta_alpha)
    ) # ∆' = Arcsin(sinδ − y * sin ξ * cos ∆α)


    return degrees(delta_alpha), degrees(delta_p)

cdef topocentric_azimuth_angle(
    cnp.ndarray[DTYPE_t, ndim=2] phi,         # φ observer latitude
    cnp.ndarray[DTYPE_t, ndim=2] delta_p,     # δ’ topocentric sun declination
    cnp.ndarray[DTYPE_t, ndim=2] H_p,         # H’ topocentric local hour angle
    cnp.ndarray[DTYPE_t, ndim=2] P,           # P is the annual average local pressure (in millibars)
    cnp.ndarray[DTYPE_t, ndim=2] T,           # T is the annual average local temperature (in degrees Celsius)
    cnp.ndarray[DTYPE_t, ndim=2] refraction
): # type: ignore
    cdef cnp.ndarray[DTYPE_t, ndim=2] e, e0, delta_e, theta, theta0
    # - in radians
    phi = radians(phi)
    delta_p = radians(delta_p)
    H_p = radians(H_p)

    # 3.14.1
    e0 = (
        arcsin(sin(phi)* sin(delta_p) + cos(phi) * cos(delta_p) * cos(H_p))
    ) # e0 = Arcsin(sin ϕ *sin δ'+ cos ϕ *cos δ'*cos H') 

    # - in degrees
    e0 = degrees(e0)

    # 3.14.2
    delta_e = (
        (P / 1010.0) * (283.0 / (273 + T))  * 1.02 / (60 * tan(radians(e0 + 10.3 / (e0 + 5.11))))
    ) # ∆e = (P / 1010) * (283 / (273 + T)) * 1.02 / (60 * tan(radians(e0 + 10.3 / (e0 + 5.11))))
    # Note that ∆e = 0 when the sun is below the horizon.
    delta_e *= e0 >= -1.0 * (0.26667 + refraction)

    # 3.14.3. Calculate the topocentric elevation angle, e (in degrees), 
    e =  e0 + delta_e       # e = e0 + ∆e

    # 3.14.4. Calculate the topocentric zenith angle, 2 (in degrees),
    theta = 90 - e          # θ = 90 − e
    theta0 = 90 - e0        # θ0 = 90 − e0

    return e, e0, theta, theta0 

cdef topocentric_astronomers_azimuth(
    cnp.ndarray[DTYPE_t, ndim=2] topocentric_local_hour_angle,
    cnp.ndarray[DTYPE_t, ndim=2] topocentric_sun_declination,
    cnp.ndarray[DTYPE_t, ndim=2] observer_latitude
): # type: ignore
    # cdef double topocentric_local_hour_angle_rad, phi, gamma
    topocentric_local_hour_angle_rad = radians(topocentric_local_hour_angle)
    phi = radians(observer_latitude)
    gamma = arctan2(
        sin(topocentric_local_hour_angle_rad), 
        cos(topocentric_local_hour_angle_rad) * sin(phi) - tan(radians(topocentric_sun_declination)) * cos(phi)
    )
    
    return degrees(gamma) % 360

cdef c_main(
    # int num_t,
    double[:, :] tc,
    cnp.ndarray[DTYPE_t, ndim=2] lat,
    cnp.ndarray[DTYPE_t, ndim=2] lon,
    cnp.ndarray[DTYPE_t, ndim=2] E,
    cnp.ndarray[DTYPE_t, ndim=2] P,
    cnp.ndarray[DTYPE_t, ndim=2] T,
    cnp.ndarray[DTYPE_t, ndim=2] refct,
    int num_threads
):
    cdef int i, n
    cdef double v, xi, delta, alpha
    cdef cnp.ndarray[DTYPE_t, ndim=2] H, H_p, delta_alpha, delta_p#, e0#, e
    cdef double[:, : , :] out
    
    n = tc.shape[1]
    out = np.empty((5, n, lat.shape[1]), dtype=DTYPE)
    for i in range(n):
        v       = tc[APARENT_SIDEREAL_TIME, i]                                  # ν
        xi      = tc[EQUATOIRAL_HORIZONAL_PARALAX, i]                           # ξ
        delta   = tc[TOPOCENTRIC_DECLINATION, i]                                # δ
        alpha   = tc[TOPOCENTRIC_RIGHT_ASCENSION, i]                            # α

        # - 3.11. Calculate the observer local hour angle, H (in degrees):
        H = (v + lon - alpha) % 360
        (
            delta_alpha, delta_p
        ) = topocentric_parallax_right_ascension_and_declination(delta, H, E, lat, xi)

        H_p = H - delta_alpha                                                   # H' = H − ∆α

        # 3.14.1
        (
            e, e0, theta, theta0                                                               # e, e0
        ) = topocentric_azimuth_angle(lat, delta_p, H_p, P, T,  refct)
        print(e, e0, theta, theta0)

        gamma = topocentric_astronomers_azimuth(H_p, delta_p, lat)
        
        
        out[0, i, ...] = 90 - e                                                 # theta[apparent zenith angle]
        out[1, i, ...] = 90 - e0                                                # theta0[zenith angle]
        out[2, i, ...] = e                                                      # epsilon[elevation angle]
        out[3, i, ...] = e0                                                     # epsilon0[apparent elevation angle]
        out[4, i, ...] = (gamma + 180) % 360                                    # gamma[azimuth angle]





    return out

def main(
    datetime_like: ArrayLike,
    latitude: ArrayLike,
    longitude: ArrayLike,
    elevation: ArrayLike | None = None,
    pressure: ArrayLike | None = None,
    temperature: ArrayLike | None = None,
    refraction: ArrayLike | None = None,
    apply_correction = False,
    int num_threads = 1,
):
    
    latitude = np.asfarray(latitude, dtype=np.float64)
    longitude = np.asfarray(longitude, dtype=np.float64)
    time_components = fastspa.time_components(datetime_like)
    return c_main(
        time_components,
        latitude,
        longitude,
        np.full_like(latitude, 0.0),
        np.full_like(latitude, 1013.25),
        np.full_like(latitude, 12.0),
        np.full_like(latitude, 0.0),
        num_threads
    )



import fastspa
import pvlib.spa
time = ["2022-01-01"]
lats = [[45.0]]
lons = [[0.0]]
dt = np.array(["2022-01-01"]).astype("datetime64[ns]")
ut = dt.astype("int64") / 1e9
delta_t = fastspa.pe4dt(dt)
elevation = np.zeros_like(lats)
print(
    np.asarray(main(dt,lats,lons, elevation)).flatten(),
    fastspa.fast_spa(dt,lats,lons).flatten(),
    pvlib.spa.solar_position(   
        ut,
        [45.0],
        [0.0],
        0.0,
        1013.25,
        temp=12,
        delta_t=delta_t,
        atmos_refract=0,
).flatten()[:-1],
    # [158.0107302 , 158.0107302 , -68.0107302 , -68.0107302 ,357.97091567],
    sep='\n'
)
# array([158.0107302 , 158.0107302 , -68.0107302 , -68.0107302 ,357.97091567])