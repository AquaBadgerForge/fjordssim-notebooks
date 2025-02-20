import numpy as np
import xesmf as xe
from scipy.interpolate import interp1d


def tranform_to_z(ds):
    """
    Transforms s coordingate to z with Vtransform = 2
    """
    zo_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
    z_rho = ds.zeta + (ds.zeta + ds.h) * zo_rho
    return z_rho.transpose()


def regrid_from_s_to_depths(values, depths, target_depths):
    """
    Args:
        values: values to be interpolated
        depths: depths of original values
        target_depths: interpolation target depths
    Returns:
        interpolated_values: result values on target depths
    """
    interpolated_shape = list(depths.shape)
    interpolated_shape[1] = len(target_depths)
    interpolated_values = np.empty(interpolated_shape)

    T, D, X, Y = values.shape
    for t in range(T):
        for x in range(X):
            for y in range(Y):
                f = interp1d(
                    depths[t, :, x, y],
                    values[t, :, x, y],
                    kind="linear",
                    bounds_error=False,
                )
                interpolated_values[t, :, x, y] = f(target_depths)

    return interpolated_values


def regrid_from_roms(ds_in, ds_out_c, ds_out_u, ds_out_v, target_depths):
    """
    Regrids roms output file 'ds_in' to target lons, lats, depths.
    Target lons and lats are in ds_out_*, for example

    ds_out_c = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88), {"units": "degrees_east"}),
        }
    )
    ds_out_u = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88 + 1), {"units": "degrees_east"}),
        }
    )
    ds_out_v = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490 + 1), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88), {"units": "degrees_east"}),
        }
    )

    """
    ds_in["z_rho"] = tranform_to_z(ds_in)

    regridder_rho = xe.Regridder(
        ds_in.rename({"lon_rho": "lon", "lat_rho": "lat"}), ds_out_c, "bilinear", unmapped_to_nan=True
    )
    regridder_u = xe.Regridder(
        ds_in.rename({"lon_u": "lon", "lat_u": "lat"}), ds_out_u, "bilinear", unmapped_to_nan=True
    )
    regridder_v = xe.Regridder(
        ds_in.rename({"lon_v": "lon", "lat_v": "lat"}), ds_out_v, "bilinear", unmapped_to_nan=True
    )

    da_temp = regridder_rho(ds_in["temp"])
    da_salt = regridder_rho(ds_in["salt"])
    da_zrho = regridder_rho(ds_in["z_rho"])
    da_u = regridder_u(ds_in["u"])
    da_v = regridder_v(ds_in["v"])

    zrho = da_zrho.values
    zrho = np.transpose(zrho, (1, 0, 2, 3))

    np_temp = regrid_from_s_to_depths(da_temp.values, zrho, target_depths)
    np_salt = regrid_from_s_to_depths(da_salt.values, zrho, target_depths)

    zu = np.zeros_like(da_u)
    zu[:, :, :, :-1] = zrho
    zu[:, :, :, -1] = zu[:, :, :, -2]
    zv = np.zeros_like(da_v)
    zv[:, :, :-1, :] = zrho
    zv[:, :, -1, :] = zv[:, :, -2, :]

    np_u = regrid_from_s_to_depths(da_u.values, zu, target_depths)
    np_v = regrid_from_s_to_depths(da_v.values, zv, target_depths)

    np_time = ds_in.ocean_time.values

    return np_time, np_temp, np_salt, np_u, np_v
