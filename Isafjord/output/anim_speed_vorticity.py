import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import contextily as ctx
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import geopandas as gpd
import colormaps as cmaps

# Load dataset with lazy loading (don't load into memory)
ds_all = xr.open_dataset('C:/Users/fisa/Documents/Isafjord/output/3dec_T5_S34.5/snapshots_ocean_3dec.nc', decode_timedelta=True).isel(z_aac=slice(0, 28))
print("Dataset loaded successfully")

# Don't slice or mask until needed - keep lazy evaluation

# Load required vars
time = ds_all.time.values
nt = len(time)  # Get total number of timesteps

# Coordinates
xC, yC = ds_all['λ_caa'], ds_all['φ_aca']  # Cell centers
xF, yF = ds_all['λ_faa'], ds_all['φ_afa']  # Cell faces
z_levels = ds_all['z_aac'].values  # Vertical levels

# Grid spacing (meters) - read from NetCDF
dx = 181.12762
dy = 222.38985

# Create meshgrids
xCm, yCm = np.meshgrid(xC, yC)
xFm, yFm = np.meshgrid(xF, yF)

# Read transect line from shapefile
gdf = gpd.read_file("C:/Users/fisa/Documents/fjordssim-notebooks/Isafjord/input/grid/02_modified_bathy/v2_210x256/transects/main/transect_main.shp")
transect_coords = list(gdf.geometry.iloc[0].coords[:])  # get all vertices

# Extract lon, lat from shapefile
lon_orig = np.array([coord[0] for coord in transect_coords])
lat_orig = np.array([coord[1] for coord in transect_coords])

# Interpolate transect line to get 100 points
distance_along = np.cumsum(np.sqrt(np.diff(lon_orig)**2 + np.diff(lat_orig)**2))
distance_along = np.insert(distance_along, 0, 0)
interp_func_lon = interp1d(distance_along, lon_orig, kind='linear')
interp_func_lat = interp1d(distance_along, lat_orig, kind='linear')

distance_interp = np.linspace(0, distance_along[-1], 100)
lon_transect = interp_func_lon(distance_interp)
lat_transect = interp_func_lat(distance_interp)

# Convert lat/lon to grid indices for sampling
# Find nearest grid points for each transect point
xs_interp = np.zeros(len(lon_transect), dtype=int)
ys_interp = np.zeros(len(lat_transect), dtype=int)
for i, (lon, lat) in enumerate(zip(lon_transect, lat_transect)):
    xs_interp[i] = np.argmin(np.abs(xC.values - lon))
    ys_interp[i] = np.argmin(np.abs(yC.values - lat))

# Color scale ranges
vminmax = {}
vminmax['speed'] = (0, 0.5)  # m/s
vminmax['vorticity'] = (-1e-3, 1e-3)  # 1/s

# Setup figure and axes - 2 rows, 2 columns
# Row 1: Speed map, Speed transect
# Row 2: Vorticity map, Vorticity transect
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

axs = {
    'speed_map': fig.add_subplot(gs[0, 0]),
    'speed_transect': fig.add_subplot(gs[0, 1]),
    'vort_map': fig.add_subplot(gs[1, 0]),
    'vort_transect': fig.add_subplot(gs[1, 1]),
}

plots = {}
colorbars = {}
transect_lines = {}


def calculate_speed(u, v):
    """Calculate speed from u and v components
    
    Handles different array shapes by finding common dimensions
    """
    # Find the minimum dimensions to handle staggered grid
    min_shape = (min(u.shape[0], v.shape[0]), min(u.shape[1], v.shape[1]))
    
    # Trim both arrays to common shape
    u_trimmed = u[:min_shape[0], :min_shape[1]]
    v_trimmed = v[:min_shape[0], :min_shape[1]]
    
    speed = np.sqrt(u_trimmed**2 + v_trimmed**2)
    return speed


def calculate_vorticity(u, v, dx, dy):
    """Calculate vertical vorticity (dv/dx - du/dy)
    
    dx and dy can be either scalars or 2D arrays matching the grid
    """
    # If dx and dy are 2D arrays, we need to compute gradients manually
    if isinstance(dx, np.ndarray) and dx.ndim == 2:
        # Manual gradient calculation for non-uniform grid
        # dv/dx
        dv_dx = np.zeros_like(v)
        dv_dx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (dx[:, 1:-1] + dx[:, 2:])
        dv_dx[:, 0] = (v[:, 1] - v[:, 0]) / dx[:, 0]
        dv_dx[:, -1] = (v[:, -1] - v[:, -2]) / dx[:, -1]
        
        # du/dy
        du_dy = np.zeros_like(u)
        du_dy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (dy[1:-1, :] + dy[2:, :])
        du_dy[0, :] = (u[1, :] - u[0, :]) / dy[0, :]
        du_dy[-1, :] = (u[-1, :] - u[-2, :]) / dy[-1, :]
        
        vorticity = dv_dx[:-1, :] - du_dy[:, :-1]
        vorticity[v[:-1, :] == 0] = np.nan
    else:
        # Uniform grid - use np.gradient
        dv_dx = np.gradient(v, dx, axis=1)
        du_dy = np.gradient(u, dy, axis=0)
        vorticity = dv_dx[:-1, :] - du_dy[:, :-1]
        vorticity[v[:-1, :] == 0] = np.nan
    
    return vorticity


def init():
    # Load only first timestep and only needed data
    ds_t0 = ds_all.isel(time=0)
    
    # Get u and v components at surface
    u_surf = ds_t0['u'].isel(z_aac=27).values
    v_surf = ds_t0['v'].isel(z_aac=27).values
    u_surf = np.where(u_surf == 0, np.nan, u_surf)
    v_surf = np.where(v_surf == 0, np.nan, v_surf)
    
    # Calculate speed at surface
    speed_surf = calculate_speed(u_surf, v_surf)
    
    # Calculate vorticity at surface (using global dx, dy)
    vort_surf = calculate_vorticity(u_surf, v_surf, dx, dy)
    
    # Speed map
    ax = axs['speed_map']
    plots['speed_map'] = ax.pcolormesh(xCm, yCm, speed_surf, cmap=cmaps.WhiteBlueGreenYellowRed,
                                       vmin=vminmax['speed'][0], vmax=vminmax['speed'][1], shading='auto')
    ax.set_title('Speed (Surface)')
    ax.grid(True)
    ctx.add_basemap(ax, crs='EPSG:4326', attribution_size=6)
    colorbars['speed_map'] = fig.colorbar(plots['speed_map'], ax=ax, label='Speed (m/s)')
    # Add transect line to map
    transect_lines['speed'] = ax.plot(lon_transect, lat_transect, 'r-', linewidth=2, label='Transect')[0]
    ax.legend()
    
    # Speed transect
    ax = axs['speed_transect']
    u_transect = ds_t0['u'].values[:, ys_interp, xs_interp]
    v_transect = ds_t0['v'].values[:, ys_interp, xs_interp]
    u_transect = np.where(u_transect == 0, np.nan, u_transect)
    v_transect = np.where(v_transect == 0, np.nan, v_transect)
    speed_transect = calculate_speed(u_transect, v_transect)
    
    X_grid, Z_grid = np.meshgrid(np.arange(len(xs_interp)), z_levels)
    speed_levels = np.linspace(vminmax['speed'][0], vminmax['speed'][1], 41)
    plots['speed_transect'] = ax.contourf(X_grid, Z_grid, speed_transect, levels=speed_levels, cmap=cmaps.WhiteBlueGreenYellowRed,
                                          extend='both')
    ax.set_title('Speed Transect')
    ax.set_xlabel('Transect Distance (index)')
    ax.set_ylabel('Depth (m)')
    ax.grid(True, alpha=0.3)
    colorbars['speed_transect'] = fig.colorbar(plots['speed_transect'], ax=ax, label='Speed (m/s)')
    
    # Vorticity map
    ax = axs['vort_map']
    plots['vort_map'] = ax.pcolormesh(xCm, yCm, vort_surf, cmap=cmaps.vik,
                                      vmin=vminmax['vorticity'][0], vmax=vminmax['vorticity'][1], shading='auto')
    ax.set_title('Vorticity (Surface)')
    ax.grid(True)
    ctx.add_basemap(ax, crs='EPSG:4326', attribution_size=6)
    colorbars['vort_map'] = fig.colorbar(plots['vort_map'], ax=ax, label='Vorticity (1/s)')
    # Add transect line to map
    transect_lines['vort'] = ax.plot(lon_transect, lat_transect, 'r-', linewidth=2, label='Transect')[0]
    ax.legend()
    
    # Vorticity transect
    ax = axs['vort_transect']
    # Calculate vorticity along transect for each depth level
    vort_transect = np.zeros_like(speed_transect)
    for k in range(len(z_levels)):
        u_level = ds_t0['u'].isel(z_aac=k).values
        v_level = ds_t0['v'].isel(z_aac=k).values
        u_level = np.where(u_level == 0, np.nan, u_level)
        v_level = np.where(v_level == 0, np.nan, v_level)
        vort_level = calculate_vorticity(u_level, v_level, dx, dy)
        vort_transect[k, :] = vort_level[ys_interp, xs_interp]
    
    vort_levels = np.linspace(vminmax['vorticity'][0], vminmax['vorticity'][1], 41)
    plots['vort_transect'] = ax.contourf(X_grid, Z_grid, vort_transect, levels=vort_levels, cmap='RdBu_r',
                                         extend='both')
    ax.set_title('Vorticity Transect')
    ax.set_xlabel('Transect Distance (index)')
    ax.set_ylabel('Depth (m)')
    ax.grid(True, alpha=0.3)
    colorbars['vort_transect'] = fig.colorbar(plots['vort_transect'], ax=ax, label='Vorticity (1/s)')
    
    return list(plots.values())


def update(i):
    # Load only current timestep
    ds_t = ds_all.isel(time=i)
    
    # Get u and v components at surface
    u_surf = ds_t['u'].isel(z_aac=27).values
    v_surf = ds_t['v'].isel(z_aac=27).values
    u_surf = np.where(u_surf == 0, np.nan, u_surf)
    v_surf = np.where(v_surf == 0, np.nan, v_surf)
    
    # Calculate speed at surface
    speed_surf = calculate_speed(u_surf, v_surf)
    
    # Calculate vorticity at surface (using global dx, dy)
    vort_surf = calculate_vorticity(u_surf, v_surf, dx, dy)
    
    # Update Speed map
    plots['speed_map'].set_array(speed_surf.ravel())
    
    # Update Speed transect
    ax = axs['speed_transect']
    ax.clear()
    u_transect = ds_t['u'].values[:, ys_interp, xs_interp]
    v_transect = ds_t['v'].values[:, ys_interp, xs_interp]
    u_transect = np.where(u_transect == 0, np.nan, u_transect)
    v_transect = np.where(v_transect == 0, np.nan, v_transect)
    speed_transect = calculate_speed(u_transect, v_transect)
    
    X_grid, Z_grid = np.meshgrid(np.arange(len(xs_interp)), z_levels)
    speed_levels = np.linspace(vminmax['speed'][0], vminmax['speed'][1], 21)
    plots['speed_transect'] = ax.contourf(X_grid, Z_grid, speed_transect, levels=speed_levels, cmap=cmaps.WhiteBlueGreenYellowRed,
                                          extend='both')
    ax.set_title('Speed Transect')
    ax.set_xlabel('Transect Distance (index)')
    ax.set_ylabel('Depth (m)')
    ax.grid(True, alpha=0.3)
    
    # Update Vorticity map
    plots['vort_map'].set_array(vort_surf.ravel())
    
    # Update Vorticity transect
    ax = axs['vort_transect']
    ax.clear()
    vort_transect = np.zeros_like(speed_transect)
    for k in range(len(z_levels)):
        u_level = ds_t['u'].isel(z_aac=k).values
        v_level = ds_t['v'].isel(z_aac=k).values
        u_level = np.where(u_level == 0, np.nan, u_level)
        v_level = np.where(v_level == 0, np.nan, v_level)
        vort_level = calculate_vorticity(u_level, v_level, dx, dy)
        vort_transect[k, :] = vort_level[ys_interp, xs_interp]
    
    vort_levels = np.linspace(vminmax['vorticity'][0], vminmax['vorticity'][1], 21)
    plots['vort_transect'] = ax.contourf(X_grid, Z_grid, vort_transect, levels=vort_levels, cmap=cmaps.vik,
                                         extend='both')
    ax.set_title('Vorticity Transect')
    ax.set_xlabel('Transect Distance (index)')
    ax.set_ylabel('Depth (m)')
    ax.grid(True, alpha=0.3)

    # Convert time to hours/days for display
    time_delta = pd.Timedelta(time[i])
    hours = time_delta.total_seconds() / 3600
    days = hours / 24
    fig.suptitle(f"Time: {days:.2f} days ({hours:.1f} hours) from start", fontsize=16)
    print(f"Frame {i}/{nt} updated")
    return list(plots.values())


# Animate
anim = FuncAnimation(fig, update, init_func=init, frames=range(0, nt, 1), blit=False)

# Save as MP4
writer = FFMpegWriter(fps=3, metadata=dict(artist='Python'), bitrate=3000)
anim.save("visual/exp_3dec_1year/anim_speed_vorticity.mp4", writer=writer)
print(f"Animation saved! Total frames: {nt}")

plt.close()
