import numpy as np
import pvlib.spa
import fast_spa


def slow_spa(
    obj,
    lat,
    lon,
    elevation=0.0,
    pressure=1013.25,
    temp=12.0,
    refraction=0.5667,
):
    delta_t = fast_spa.pe4dt(obj)
    dt = np.asanyarray(obj, dtype="datetime64[ns]")
    unix_time = dt.astype(np.float64) // 1e9

    x = np.stack(
        [
            np.stack(
                pvlib.spa.solar_position_numpy(
                    ut,
                    lat=lat,
                    lon=lon,
                    elev=elevation,
                    pressure=pressure,
                    temp=temp,
                    delta_t=delta_t[i : i + 1],
                    atmos_refract=refraction,
                    numthreads=None,
                    sst=False,
                    esd=False,
                )[:-1]
            )
            for i, ut in enumerate(unix_time)
        ],
        axis=1,
    )
    return x


def slow_jme(obj):
    dt = np.asanyarray(obj, dtype="datetime64[ns]")
    year = dt.astype("datetime64[Y]").astype(int) + 1970
    month = dt.astype("datetime64[M]").astype(int) % 12 + 1
    unix_time = dt.astype(np.float64) // 1e9
    delta_t = pvlib.spa.calculate_deltat(year=year, month=month)

    jd = pvlib.spa.julian_day(unix_time)
    jde = pvlib.spa.julian_ephemeris_day(jd, delta_t=delta_t)
    jce = pvlib.spa.julian_ephemeris_century(jde)
    jme = pvlib.spa.julian_ephemeris_millennium(jce)

    return jme
