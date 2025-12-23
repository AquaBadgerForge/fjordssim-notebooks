"""
Create Initial Conditions for Isafjord Simulation
==================================================

This script creates initial condition NetCDF files for T and S based on 
CTD profile data from cruise B3-2016.

Method:
1. Load CTD profiles from B3-2016 cruise
2. Interpolate profiles to model vertical grid (z_centers)
3. Use nearest neighbor interpolation to fill 3D grid
4. Apply Gaussian smoothing
5. Save as NetCDF compatible with FjordsSim.jl

Author: Generated for FjordsSim.jl
Date: 2025-12-03
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')

# =============================================================================
# Configuration
# =============================================================================
PROFILE_DB_PATH = Path(r"C:\Users\fisa\Documents\Isafjord\validation\profile_database")
BATHY_PATH = Path(r"C:\Users\fisa\Documents\Isafjord\input\Isafjord_bathymetry_210x256.nc")
TRANSECT_PATH = Path(r"C:\Users\fisa\Documents\fjordssim-notebooks\Isafjord\input\grid\02_modified_bathy\v2_210x256\transects\main\transect_main.shp")
OUTPUT_DIR = Path(r"C:\Users\fisa\Documents\fjordssim-notebooks\Isafjord\input\ini")
OUTPUT_FILE = OUTPUT_DIR / "Isf_ini_210x256_B3-2016.nc"

CRUISE_NAME = "B3-2016"
SMOOTHING_SIGMA = (1, 7, 7)  # (z, y, x) - large-scale smoothing

# =============================================================================
# 1. Load Bathymetry Grid
# =============================================================================
print("=" * 60)
print("1. LOADING BATHYMETRY GRID")
print("=" * 60)

bathy = xr.open_dataset(BATHY_PATH)
print(f"Grid dimensions: {bathy.dims}")

# Extract grid parameters
h = bathy['h'].values  # Negative values for depth
lat_centers = bathy['lat'].values
lon_centers = bathy['lon'].values
z_faces = bathy['z_faces'].values

# Grid dimensions
nx = len(lon_centers)
ny = len(lat_centers)
nz = len(z_faces) - 1

# Calculate z_centers
z_centers = np.array([(z_faces[i] + z_faces[i+1]) / 2 for i in range(len(z_faces) - 1)])

# Create wet mask (where h < 0, i.e., water)
wet_mask = h < 0
grid_depth = -h  # Make positive for visualization

print(f"Grid size: {nx} x {ny} x {nz}")
print(f"Latitude range: {lat_centers.min():.4f}°N to {lat_centers.max():.4f}°N")
print(f"Longitude range: {lon_centers.min():.4f}°E to {lon_centers.max():.4f}°E")
print(f"z_centers range: {z_centers[0]:.1f}m to {z_centers[-1]:.1f}m")
print(f"Wet cells: {np.sum(wet_mask)} ({100*np.sum(wet_mask)/(nx*ny):.1f}%)")

# =============================================================================
# 2. Load CTD Profiles from B3-2016
# =============================================================================
print("\n" + "=" * 60)
print(f"2. LOADING CTD PROFILES FROM {CRUISE_NAME}")
print("=" * 60)

# Load index
index = pd.read_csv(PROFILE_DB_PATH / 'index.csv')
cruise_profiles = index[index['cruise'] == CRUISE_NAME].copy()

print(f"Found {len(cruise_profiles)} profiles from {CRUISE_NAME}")
print(f"Date range: {cruise_profiles['date'].min()} to {cruise_profiles['date'].max()}")

# Load all profile data
profiles_data = []
for _, row in cruise_profiles.iterrows():
    filepath = PROFILE_DB_PATH / row['filepath']
    df = pd.read_csv(filepath, comment='#')
    df['latitude'] = row['latitude']
    df['longitude'] = row['longitude']
    df['station'] = row['station']
    df['profile_id'] = row['profile_id']
    profiles_data.append(df)

all_profiles = pd.concat(profiles_data, ignore_index=True)
print(f"Total data points: {len(all_profiles)}")
print(f"Temperature range: {all_profiles['temperature_degC'].min():.2f} to {all_profiles['temperature_degC'].max():.2f} °C")
print(f"Salinity range: {all_profiles['salinity'].min():.2f} to {all_profiles['salinity'].max():.2f}")

# Get unique station locations
stations = cruise_profiles[['latitude', 'longitude', 'station', 'profile_id']].drop_duplicates()
print(f"\nUnique stations: {len(stations)}")

# =============================================================================
# 3. Interpolate Profiles to Model Vertical Grid
# =============================================================================
print("\n" + "=" * 60)
print("3. INTERPOLATING PROFILES TO MODEL VERTICAL GRID")
print("=" * 60)

# For each profile, interpolate to z_centers
profile_T_interp = []
profile_S_interp = []
profile_lats = []
profile_lons = []
profile_ids = []

for profile_id in cruise_profiles['profile_id'].values:
    prof_data = all_profiles[all_profiles['profile_id'] == profile_id]
    
    # Get profile location
    lat = prof_data['latitude'].iloc[0]
    lon = prof_data['longitude'].iloc[0]
    
    # Get depth (negative for interpolation)
    depth = -prof_data['depth_m'].values  # Make negative to match z_centers
    temp = prof_data['temperature_degC'].values
    sal = prof_data['salinity'].values
    
    # Sort by depth (from surface to deep)
    sort_idx = np.argsort(depth)[::-1]
    depth = depth[sort_idx]
    temp = temp[sort_idx]
    sal = sal[sort_idx]
    
    # Interpolate to z_centers
    # Only interpolate within the profile depth range
    valid_z = (z_centers >= depth.min()) & (z_centers <= depth.max())
    
    T_interp = np.full(nz, np.nan)
    S_interp = np.full(nz, np.nan)
    
    if np.sum(valid_z) > 0:
        try:
            f_T = interp1d(depth, temp, kind='linear', bounds_error=False, fill_value=np.nan)
            f_S = interp1d(depth, sal, kind='linear', bounds_error=False, fill_value=np.nan)
            
            T_interp[valid_z] = f_T(z_centers[valid_z])
            S_interp[valid_z] = f_S(z_centers[valid_z])
            
            # Extend surface values upward
            first_valid = np.where(~np.isnan(T_interp))[0]
            if len(first_valid) > 0:
                first_idx = first_valid[-1]  # Shallowest valid
                T_interp[first_idx:] = T_interp[first_idx]
                S_interp[first_idx:] = S_interp[first_idx]
            
            # Extend bottom values downward
            last_valid = np.where(~np.isnan(T_interp))[0]
            if len(last_valid) > 0:
                last_idx = last_valid[0]  # Deepest valid
                T_interp[:last_idx+1] = T_interp[last_idx]
                S_interp[:last_idx+1] = S_interp[last_idx]
                
        except Exception as e:
            print(f"  Warning: Could not interpolate {profile_id}: {e}")
            continue
    
    profile_T_interp.append(T_interp)
    profile_S_interp.append(S_interp)
    profile_lats.append(lat)
    profile_lons.append(lon)
    profile_ids.append(profile_id)

profile_T_interp = np.array(profile_T_interp)  # Shape: (n_profiles, nz)
profile_S_interp = np.array(profile_S_interp)
profile_lats = np.array(profile_lats)
profile_lons = np.array(profile_lons)

print(f"Successfully interpolated {len(profile_lats)} profiles")
print(f"T interpolated range: {np.nanmin(profile_T_interp):.2f} to {np.nanmax(profile_T_interp):.2f} °C")
print(f"S interpolated range: {np.nanmin(profile_S_interp):.2f} to {np.nanmax(profile_S_interp):.2f}")

# =============================================================================
# 4. Nearest Neighbor Interpolation to 3D Grid
# =============================================================================
print("\n" + "=" * 60)
print("4. NEAREST NEIGHBOR INTERPOLATION TO 3D GRID")
print("=" * 60)

# Initialize 3D arrays
T_3d = np.full((nz, ny, nx), np.nan, dtype=np.float32)
S_3d = np.full((nz, ny, nx), np.nan, dtype=np.float32)

# Create 2D lon/lat grids
lon_grid_2d, lat_grid_2d = np.meshgrid(lon_centers, lat_centers)

# Build KD-tree from profile locations
profile_coords = np.column_stack([profile_lats, profile_lons])
tree = cKDTree(profile_coords)

# For each wet grid cell, find nearest profile and assign values
print("Filling 3D grid with nearest neighbor interpolation...")

for j in range(ny):
    for i in range(nx):
        if wet_mask[j, i]:
            # Find nearest profile
            lat_cell = lat_centers[j]
            lon_cell = lon_centers[i]
            dist, idx = tree.query([lat_cell, lon_cell])
            
            # Assign T and S from nearest profile
            T_3d[:, j, i] = profile_T_interp[idx, :]
            S_3d[:, j, i] = profile_S_interp[idx, :]

# Count valid cells
valid_T = np.sum(~np.isnan(T_3d))
valid_S = np.sum(~np.isnan(S_3d))
print(f"Valid T cells: {valid_T} ({100*valid_T/(nz*ny*nx):.1f}%)")
print(f"Valid S cells: {valid_S} ({100*valid_S/(nz*ny*nx):.1f}%)")

# Store original for comparison
T_3d_original = T_3d.copy()
S_3d_original = S_3d.copy()

# =============================================================================
# 5. Apply Gaussian Smoothing
# =============================================================================
print("\n" + "=" * 60)
print("5. APPLYING GAUSSIAN SMOOTHING")
print("=" * 60)
print(f"Smoothing sigma (z, y, x): {SMOOTHING_SIGMA}")

# Replace NaN with mean for smoothing
T_mean = np.nanmean(T_3d)
S_mean = np.nanmean(S_3d)

T_filled = np.where(np.isnan(T_3d), T_mean, T_3d)
S_filled = np.where(np.isnan(S_3d), S_mean, S_3d)

# Apply Gaussian smoothing
T_smoothed = gaussian_filter(T_filled, sigma=SMOOTHING_SIGMA)
S_smoothed = gaussian_filter(S_filled, sigma=SMOOTHING_SIGMA)

# Restore land mask (set to 0 for land cells)
T_final = np.where(wet_mask[np.newaxis, :, :], T_smoothed, 0.0)
S_final = np.where(wet_mask[np.newaxis, :, :], S_smoothed, 0.0)

print(f"T final range: {T_final[T_final != 0].min():.2f} to {T_final[T_final != 0].max():.2f} °C")
print(f"S final range: {S_final[S_final != 0].min():.2f} to {S_final[S_final != 0].max():.2f}")

# =============================================================================
# 6. Load Transect for Visualization
# =============================================================================
print("\n" + "=" * 60)
print("6. LOADING TRANSECT FOR VISUALIZATION")
print("=" * 60)

# Read transect line from shapefile
gdf = gpd.read_file(TRANSECT_PATH)
transect_coords = list(gdf.geometry.iloc[0].coords[:])

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

# Convert lat/lon to grid indices
xs_interp = np.zeros(len(lon_transect), dtype=int)
ys_interp = np.zeros(len(lat_transect), dtype=int)
for i, (lon, lat) in enumerate(zip(lon_transect, lat_transect)):
    xs_interp[i] = np.argmin(np.abs(lon_centers - lon))
    ys_interp[i] = np.argmin(np.abs(lat_centers - lat))

# Calculate distance along transect in km
R_earth = 6371  # km
dist_km = np.zeros(len(lon_transect))
for i in range(1, len(lon_transect)):
    dlat = np.radians(lat_transect[i] - lat_transect[i-1])
    dlon = np.radians(lon_transect[i] - lon_transect[i-1])
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat_transect[i-1])) * np.cos(np.radians(lat_transect[i])) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dist_km[i] = dist_km[i-1] + R_earth * c

print(f"Transect length: {dist_km[-1]:.1f} km")
print(f"Transect points: {len(lon_transect)}")

# =============================================================================
# 7. Plot Maps and Transects
# =============================================================================
print("\n" + "=" * 60)
print("7. PLOTTING MAPS AND TRANSECTS")
print("=" * 60)

# Surface level index (shallowest)
surf_idx = -1  # Last index is surface

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.subplots_adjust(hspace=0.3, wspace=0.25)

# Pre-compute masked surface data
T_surf = np.where(T_final[surf_idx, :, :] == 0, np.nan, T_final[surf_idx, :, :])
S_surf = np.where(S_final[surf_idx, :, :] == 0, np.nan, S_final[surf_idx, :, :])

# Temperature map - use imshow (faster than pcolormesh)
ax1 = axes[0, 0]
extent = [lon_centers[0], lon_centers[-1], lat_centers[0], lat_centers[-1]]
im1 = ax1.imshow(T_surf, cmap='RdYlBu_r', origin='lower', extent=extent, aspect='auto')
ax1.plot(lon_transect, lat_transect, 'k-', linewidth=2, label='Transect')
ax1.scatter(profile_lons, profile_lats, c='lime', s=50, edgecolors='black', zorder=10, label='CTD Stations')
plt.colorbar(im1, ax=ax1, label='Temperature (°C)')
ax1.set_title(f'Surface Temperature (z = {z_centers[surf_idx]:.1f}m)')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.legend(loc='lower left')

# Salinity map - use imshow
ax2 = axes[0, 1]
im2 = ax2.imshow(S_surf, cmap='viridis', origin='lower', extent=extent, aspect='auto')
ax2.plot(lon_transect, lat_transect, 'k-', linewidth=2, label='Transect')
ax2.scatter(profile_lons, profile_lats, c='lime', s=50, edgecolors='black', zorder=10, label='CTD Stations')
plt.colorbar(im2, ax=ax2, label='Salinity (PSU)')
ax2.set_title(f'Surface Salinity (z = {z_centers[surf_idx]:.1f}m)')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.legend(loc='lower left')

# Extract transect data once
T_transect = T_final[:, ys_interp, xs_interp]
T_transect = np.where(T_transect == 0, np.nan, T_transect)
S_transect = S_final[:, ys_interp, xs_interp]
S_transect = np.where(S_transect == 0, np.nan, S_transect)

# Temperature transect - use pcolormesh (faster than contourf for this size)
ax3 = axes[1, 0]
X_grid, Z_grid = np.meshgrid(dist_km, z_centers)
cf3 = ax3.pcolormesh(X_grid, Z_grid, T_transect, cmap='RdYlBu_r', shading='auto')
plt.colorbar(cf3, ax=ax3, label='Temperature (°C)')
ax3.set_title('Temperature Transect')
ax3.set_xlabel('Distance along transect (km)')
ax3.set_ylabel('Depth (m)')
ax3.set_ylim([z_centers[0], 0])

# Salinity transect - use pcolormesh
ax4 = axes[1, 1]
cf4 = ax4.pcolormesh(X_grid, Z_grid, S_transect, cmap='viridis', shading='auto')
plt.colorbar(cf4, ax=ax4, label='Salinity (PSU)')
ax4.set_title('Salinity Transect')
ax4.set_xlabel('Distance along transect (km)')
ax4.set_ylabel('Depth (m)')
ax4.set_ylim([z_centers[0], 0])

plt.suptitle(f'Initial Conditions from {CRUISE_NAME} CTD Profiles\n(Smoothed, σ={SMOOTHING_SIGMA})', 
             fontsize=14, fontweight='bold')
plt.savefig(OUTPUT_DIR / 'ini_TS_maps_transects.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'ini_TS_maps_transects.png'}")

# =============================================================================
# 8. Plot Smoothed vs Original Profiles
# =============================================================================
print("\n" + "=" * 60)
print("8. PLOTTING SMOOTHED VS ORIGINAL PROFILES")
print("=" * 60)

# Select a subset of profiles to compare
n_compare = min(6, len(profile_ids))
fig, axes = plt.subplots(2, n_compare, figsize=(4*n_compare, 10), sharey=True)

for i, profile_id in enumerate(profile_ids[:n_compare]):
    # Get original profile data
    prof_data = all_profiles[all_profiles['profile_id'] == profile_id]
    lat = prof_data['latitude'].iloc[0]
    lon = prof_data['longitude'].iloc[0]
    
    # Find nearest grid cell
    j_idx = np.argmin(np.abs(lat_centers - lat))
    i_idx = np.argmin(np.abs(lon_centers - lon))
    
    # Original profile
    depth_orig = -prof_data['depth_m'].values
    T_orig = prof_data['temperature_degC'].values
    S_orig = prof_data['salinity'].values
    
    # Interpolated (before smoothing)
    T_interp_profile = T_3d_original[:, j_idx, i_idx]
    S_interp_profile = S_3d_original[:, j_idx, i_idx]
    
    # Smoothed
    T_smooth_profile = T_final[:, j_idx, i_idx]
    S_smooth_profile = S_final[:, j_idx, i_idx]
    
    # Temperature plot
    ax = axes[0, i]
    ax.plot(T_orig, depth_orig, 'k-', linewidth=2, label='Original', marker='o', markersize=3)
    ax.plot(T_interp_profile, z_centers, 'b--', linewidth=1.5, label='Interpolated')
    ax.plot(T_smooth_profile, z_centers, 'r-', linewidth=1.5, label='Smoothed')
    ax.set_xlabel('Temperature (°C)')
    ax.set_title(f'Station {prof_data["station"].iloc[0]}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([T_orig.min()-0.5, T_orig.max()+0.5])
    if i == 0:
        ax.set_ylabel('Depth (m)')
        ax.legend(loc='lower right', fontsize=8)
    
    # Salinity plot
    ax = axes[1, i]
    ax.plot(S_orig, depth_orig, 'k-', linewidth=2, label='Original', marker='o', markersize=3)
    ax.plot(S_interp_profile, z_centers, 'b--', linewidth=1.5, label='Interpolated')
    ax.plot(S_smooth_profile, z_centers, 'r-', linewidth=1.5, label='Smoothed')
    ax.set_xlabel('Salinity (PSU)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([S_orig.min()-0.1, S_orig.max()+0.1])
    if i == 0:
        ax.set_ylabel('Depth (m)')
        ax.legend(loc='lower right', fontsize=8)

axes[0, 0].set_ylim([z_centers[0], 0])
axes[1, 0].set_ylim([z_centers[0], 0])

plt.suptitle(f'Profile Comparison: Original vs Interpolated vs Smoothed\n{CRUISE_NAME}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ini_profile_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'ini_profile_comparison.png'}")

# =============================================================================
# 9. Create NetCDF Output File
# =============================================================================
print("\n" + "=" * 60)
print("9. CREATING NETCDF OUTPUT FILE")
print("=" * 60)

# Time coordinate (single timestep: 1991-01-01 00:00:00)
time_value = pd.Timestamp('1991-01-01 00:00:00')

# Add time dimension to T and S
T_out = T_final[np.newaxis, :, :, :]  # Shape: (1, nz, ny, nx)
S_out = S_final[np.newaxis, :, :, :]

# Compute face coordinates
def compute_faces(centers):
    """Compute face coordinates from centers"""
    spacing = np.mean(np.diff(centers))
    faces = np.concatenate([
        [centers[0] - spacing/2],
        (centers[:-1] + centers[1:]) / 2,
        [centers[-1] + spacing/2]
    ])
    return faces

lon_faces = compute_faces(lon_centers)
lat_faces = compute_faces(lat_centers)

# Create output dataset
ds_out = xr.Dataset(
    {
        "T": (["time", "Nz", "Ny", "Nx"], T_out.astype(np.float32)),
        "S": (["time", "Nz", "Ny", "Nx"], S_out.astype(np.float32)),
    },
    coords={
        "time": [time_value],
        "Nz": z_centers,
        "Ny": lat_centers,
        "Ny_faces": lat_faces,
        "Nx": lon_centers,
        "Nx_faces": lon_faces,
    },
)

# Add attributes
ds_out['T'].attrs = {
    'long_name': 'potential temperature',
    'units': 'degrees_C',
    'source': f'CTD profiles from cruise {CRUISE_NAME}',
    'interpolation': 'nearest neighbor',
    'smoothing': f'Gaussian filter sigma={SMOOTHING_SIGMA}'
}
ds_out['S'].attrs = {
    'long_name': 'salinity',
    'units': 'PSU',
    'source': f'CTD profiles from cruise {CRUISE_NAME}',
    'interpolation': 'nearest neighbor',
    'smoothing': f'Gaussian filter sigma={SMOOTHING_SIGMA}'
}

ds_out['Nz'].attrs = {'long_name': 'depth', 'units': 'm', 'positive': 'up'}
ds_out['Ny'].attrs = {'long_name': 'latitude', 'units': 'degrees_north'}
ds_out['Nx'].attrs = {'long_name': 'longitude', 'units': 'degrees_east'}

ds_out.attrs = {
    'title': f'Initial conditions for Isafjardardju (210x256) from {CRUISE_NAME}',
    'source': f'CTD profile database - cruise {CRUISE_NAME}',
    'created': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'cruise_date': cruise_profiles['date'].iloc[0],
    'n_profiles': len(profile_ids),
    'grid': 'Isafjord_bathymetry_210x256.nc',
    'smoothing_sigma': str(SMOOTHING_SIGMA),
    'description': 'Initial T and S fields for FjordsSim.jl simulation'
}

print("\n=== Output Dataset ===")
print(ds_out)

# Save to NetCDF
ds_out.to_netcdf(OUTPUT_FILE)
print(f"\n✓ Initial conditions saved to: {OUTPUT_FILE}")

import os
file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024**2)
print(f"  File size: {file_size_mb:.2f} MB")

# =============================================================================
# 10. Verification
# =============================================================================
print("\n" + "=" * 60)
print("10. VERIFICATION")
print("=" * 60)

# Reload and verify
ds_verify = xr.open_dataset(OUTPUT_FILE)
print("\n=== Verification ===")
print(f"Time: {ds_verify.time.values[0]}")
print(f"Dimensions: {ds_verify.dims}")

for var in ['T', 'S']:
    data = ds_verify[var].values
    valid_data = data[data != 0]
    print(f"{var}: min={valid_data.min():.3f}, max={valid_data.max():.3f}, mean={valid_data.mean():.3f}")

ds_verify.close()

print("\n" + "=" * 60)
print("COMPLETE!")
print("=" * 60)
print(f"Output file: {OUTPUT_FILE}")
print(f"Figures saved to: {OUTPUT_DIR}")
