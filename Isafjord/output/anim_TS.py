import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import contextily as ctx
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import geopandas as gpd

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

# surface
vminmax = {}
vminmax['T'] = (1.5, 13)
vminmax['S'] = (33.8, 34.8)

# Setup figure and axes - 2 rows, 2 columns
# Row 1: Temperature map, Temperature transect
# Row 2: Salinity map, Salinity transect
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

axs = {
    'T_map': fig.add_subplot(gs[0, 0]),
    'T_transect': fig.add_subplot(gs[0, 1]),
    'S_map': fig.add_subplot(gs[1, 0]),
    'S_transect': fig.add_subplot(gs[1, 1]),
}

plots = {}
colorbars = {}
transect_lines = {}


def init():
    # Load only first timestep and only needed data
    ds_t0 = ds_all.isel(time=0)
    
    # Temperature map - load only surface level
    ax = axs['T_map']
    xCm, yCm = np.meshgrid(xC, yC)
    T_surf = ds_t0['T'].isel(z_aac=27).values
    T_surf = np.where(T_surf == 0, np.nan, T_surf)  # Mask zeros efficiently
    plots['T_map'] = ax.pcolormesh(xCm, yCm, T_surf, cmap='RdYlBu_r',
                                   vmin=vminmax['T'][0], vmax=vminmax['T'][1])
    ax.set_title('Temperature (Surface)')
    ax.grid(True)
    ctx.add_basemap(ax, crs='EPSG:4326', attribution_size=6)
    colorbars['T_map'] = fig.colorbar(plots['T_map'], ax=ax, label='T (°C)')
    # Add transect line to map
    transect_lines['T'] = ax.plot(lon_transect, lat_transect, 'r-', linewidth=2, label='Transect')[0]
    ax.legend()
    
    # Temperature transect - load only transect points
    ax = axs['T_transect']
    T_transect = ds_t0['T'].values[:, ys_interp, xs_interp]
    T_transect = np.where(T_transect == 0, np.nan, T_transect)
    
    X_grid, Z_grid = np.meshgrid(np.arange(len(xs_interp)), z_levels)
    T_levels = np.linspace(vminmax['T'][0], vminmax['T'][1], 41)
    plots['T_transect'] = ax.contourf(X_grid, Z_grid, T_transect, levels=T_levels, cmap='RdYlBu_r',
                                      extend='both')
    ax.set_title('Temperature Transect')
    ax.set_xlabel('Transect Distance (index)')
    ax.set_ylabel('Depth (m)')
    ax.grid(True, alpha=0.3)
    colorbars['T_transect'] = fig.colorbar(plots['T_transect'], ax=ax, label='T (°C)')
    
    # Salinity map - load only surface level
    ax = axs['S_map']
    S_surf = ds_t0['S'].isel(z_aac=27).values
    S_surf = np.where(S_surf == 0, np.nan, S_surf)
    plots['S_map'] = ax.pcolormesh(xCm, yCm, S_surf, cmap='viridis',
                                   vmin=vminmax['S'][0], vmax=vminmax['S'][1])
    ax.set_title('Salinity (Surface)')
    ax.grid(True)
    ctx.add_basemap(ax, crs='EPSG:4326', attribution_size=6)
    colorbars['S_map'] = fig.colorbar(plots['S_map'], ax=ax, label='S (PSU)')
    # Add transect line to map
    transect_lines['S'] = ax.plot(lon_transect, lat_transect, 'r-', linewidth=2, label='Transect')[0]
    ax.legend()
    
    # Salinity transect - load only transect points
    ax = axs['S_transect']
    S_transect = ds_t0['S'].values[:, ys_interp, xs_interp]
    S_transect = np.where(S_transect == 0, np.nan, S_transect)
    
    S_levels = np.linspace(vminmax['S'][0], vminmax['S'][1], 41)
    plots['S_transect'] = ax.contourf(X_grid, Z_grid, S_transect, levels=S_levels, cmap='viridis',
                                      extend='both')
    ax.set_title('Salinity Transect')
    ax.set_xlabel('Transect Distance (index)')
    ax.set_ylabel('Depth (m)')
    ax.grid(True, alpha=0.3)
    colorbars['S_transect'] = fig.colorbar(plots['S_transect'], ax=ax, label='S (PSU)')
    
    return list(plots.values())

def update(i):
    # Load only current timestep - lazy loading with chunks
    ds_t = ds_all.isel(time=i)
    
    # Update Temperature map - load only surface level
    T_surf = ds_t['T'].isel(z_aac=27).values
    T_surf = np.where(T_surf == 0, np.nan, T_surf)
    plots['T_map'].set_array(T_surf.ravel())
    
    # Update Temperature transect - vectorized slicing
    ax = axs['T_transect']
    ax.clear()
    T_transect = ds_t['T'].values[:, ys_interp, xs_interp]
    T_transect = np.where(T_transect == 0, np.nan, T_transect)
    
    X_grid, Z_grid = np.meshgrid(np.arange(len(xs_interp)), z_levels)
    T_levels = np.linspace(vminmax['T'][0], vminmax['T'][1], 21)
    plots['T_transect'] = ax.contourf(X_grid, Z_grid, T_transect, levels=T_levels, cmap='RdYlBu_r',
                                      extend='both')
    ax.set_title('Temperature Transect')
    ax.set_xlabel('Transect Distance (index)')
    ax.set_ylabel('Depth (m)')
    ax.grid(True, alpha=0.3)
    
    # Update Salinity map - load only surface level
    S_surf = ds_t['S'].isel(z_aac=27).values
    S_surf = np.where(S_surf == 0, np.nan, S_surf)
    plots['S_map'].set_array(S_surf.ravel())
    
    # Update Salinity transect - vectorized slicing
    ax = axs['S_transect']
    ax.clear()
    S_transect = ds_t['S'].values[:, ys_interp, xs_interp]
    S_transect = np.where(S_transect == 0, np.nan, S_transect)
    
    S_levels = np.linspace(vminmax['S'][0], vminmax['S'][1], 21)
    plots['S_transect'] = ax.contourf(X_grid, Z_grid, S_transect, levels=S_levels, cmap='viridis',
                                      extend='both')
    ax.set_title('Salinity Transect')
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
anim = FuncAnimation(fig, update, init_func=init, frames=range(0,nt), blit=False)

# Save as MP4
writer = FFMpegWriter(fps=10, metadata=dict(artist='Python'), bitrate=3000)
anim.save("visual/exp_3dec_1year/anim_TS.mp4", writer=writer)
print(f"Animation saved! Total frames: {nt}")

plt.close()
