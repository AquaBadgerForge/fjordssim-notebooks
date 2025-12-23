"""
Compare CTD profiles from observations with model output.
Creates pairwise T and S profile comparisons for each observation,
with model data profiles as semi-transparent lines for 5 days before and after.
"""

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os
import contextily as ctx

# ============================================================================
# Configuration
# ============================================================================

# Paths
MODEL_PATH = 'C:/Users/fisa/Documents/Isafjord/output/3dec_T5_S34.5/snapshots_ocean_3dec.nc'
PROFILE_DB_PATH = Path('C:/Users/fisa/Documents/Isafjord/validation/profile_database')
OUTPUT_DIR = Path('C:/Users/fisa/Documents/fjordssim-notebooks/Isafjord/output/visual/exp_3dec_1year/single_ctds_vs_model')

# Days window around observation for model comparison
DAYS_WINDOW = 7

# Maximum distance (km) between observation and model grid point
MAX_DISTANCE_KM = 10

# ============================================================================
# Load model data
# ============================================================================

print("Loading model data...")
ds_model = xr.open_dataset(MODEL_PATH, decode_timedelta=True)
print(f"Model dataset loaded: {ds_model.dims}")

# Coordinates - cell centers
xC = ds_model['λ_caa'].values  # longitude
yC = ds_model['φ_aca'].values  # latitude
z_model = ds_model['z_aac'].values  # depth levels (negative values typically)

# Get time info - model is daily output for 1991
time_model = ds_model.time.values
nt = len(time_model)
print(f"Model has {nt} timesteps (days)")

# Convert model times to day of year (1-365/366)
# Model output is daily for 1991
model_doy = np.arange(1, nt + 1)  # Day 1 = first output, etc.

# ============================================================================
# Load profile database
# ============================================================================

print("\nLoading profile database...")
index_df = pd.read_csv(PROFILE_DB_PATH / 'index.csv')
index_df['datetime'] = pd.to_datetime(index_df['datetime'])
index_df['doy'] = index_df['datetime'].dt.dayofyear  # Day of year (1-366)
print(f"Loaded {len(index_df)} profiles")

# ============================================================================
# Helper functions
# ============================================================================

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in kilometers.
    """
    # Convert to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in kilometers
    r = 6371
    return c * r

def find_nearest_grid_point(lon, lat, xC, yC):
    """Find nearest grid indices for a given lon/lat and return distance."""
    x_idx = np.argmin(np.abs(xC - lon))
    y_idx = np.argmin(np.abs(yC - lat))
    
    # Calculate distance to nearest grid point
    distance_km = haversine_distance(lon, lat, xC[x_idx], yC[y_idx])
    
    return x_idx, y_idx, distance_km

def extract_model_profile(ds, time_idx, x_idx, y_idx):
    """Extract T and S profiles from model at given location and time."""
    T_profile = ds['T'].isel(time=time_idx).values[:, y_idx, x_idx]
    S_profile = ds['S'].isel(time=time_idx).values[:, y_idx, x_idx]
    
    # Mask zeros (land/invalid values)
    T_profile = np.where(T_profile == 0, np.nan, T_profile)
    S_profile = np.where(S_profile == 0, np.nan, S_profile)
    
    return T_profile, S_profile

def load_ctd_profile(filepath):
    """Load CTD profile data from CSV file."""
    df = pd.read_csv(filepath, comment='#')
    return df['depth_m'].values, df['temperature_degC'].values, df['salinity'].values

def get_matching_model_days(obs_doy, model_doy, days_window=5):
    """
    Find model timesteps that match +/- days_window around observation day of year.
    Ignores year - just matches by day of year.
    """
    matching_indices = []
    
    for day_offset in range(-days_window, days_window + 1):
        target_doy = obs_doy + day_offset
        
        # Handle year wrap-around
        if target_doy < 1:
            target_doy += 365
        elif target_doy > 365:
            target_doy -= 365
        
        # Find model index for this day of year
        # Model day 1 = first timestep (index 0), etc.
        if 1 <= target_doy <= len(model_doy):
            matching_indices.append((target_doy - 1, day_offset))  # (index, offset)
    
    return matching_indices

# ============================================================================
# Create output directory
# ============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\nOutput directory: {OUTPUT_DIR}")

# ============================================================================
# Generate comparison plots
# ============================================================================

print("\nGenerating comparison plots...")

skipped_distance = 0
generated_count = 0

for idx, row in index_df.iterrows():
    profile_id = row['profile_id']
    lat = row['latitude']
    lon = row['longitude']
    obs_doy = row['doy']
    obs_date = row['date']
    
    print(f"\nProcessing {idx+1}/{len(index_df)}: {profile_id}")
    print(f"  Location: ({lat:.4f}, {lon:.4f}), Day of year: {obs_doy}")
    
    # Find nearest grid point
    x_idx, y_idx, distance_km = find_nearest_grid_point(lon, lat, xC, yC)
    
    # Check distance filter
    if distance_km > MAX_DISTANCE_KM:
        print(f"  SKIPPED: Distance to nearest grid point ({distance_km:.2f} km) > {MAX_DISTANCE_KM} km")
        skipped_distance += 1
        continue
    
    print(f"  Distance to grid point: {distance_km:.2f} km")
    
    # Check if point is within model domain
    if x_idx < 0 or x_idx >= len(xC) or y_idx < 0 or y_idx >= len(yC):
        print(f"  WARNING: Point outside model domain, skipping")
        continue
    
    # Load CTD profile
    ctd_filepath = PROFILE_DB_PATH / row['filepath']
    try:
        depth_ctd, T_ctd, S_ctd = load_ctd_profile(ctd_filepath)
    except Exception as e:
        print(f"  ERROR loading CTD profile: {e}")
        continue
    
    # Get matching model days
    matching_days = get_matching_model_days(obs_doy, model_doy, DAYS_WINDOW)
    
    if not matching_days:
        print(f"  WARNING: No matching model days found, skipping")
        continue
    
    # Create figure with map on top row, profiles on bottom row
    # Layout: Row 1: [Map spanning full width]
    #         Row 2: [T profile] [S profile]
    fig = plt.figure(figsize=(8, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.25, wspace=0.25)
    
    ax_map = fig.add_subplot(gs[0, :])  # Map spanning full width (top row)
    ax1 = fig.add_subplot(gs[1, 0])  # Temperature profile (bottom left)
    ax2 = fig.add_subplot(gs[1, 1])  # Salinity profile (bottom right)
    
    # Get model grid point coordinates
    model_lon = xC[x_idx]
    model_lat = yC[y_idx]
    
    # ===== Map plot (full model domain) =====
    # Set map extent to cover full model domain
    ax_map.set_xlim(xC.min(), xC.max())
    ax_map.set_ylim(yC.min(), yC.max())
    
    # Add basemap first
    try:
        ctx.add_basemap(ax_map, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, attribution_size=5)
    except Exception as e:
        ax_map.set_facecolor('lightblue')
        print(f"  Warning: Could not add basemap: {e}")
    
    # Plot observation point (red) and model grid point (blue)
    ax_map.scatter(lon, lat, c='tomato', s=150, marker='o', label='CTD obs', zorder=5, edgecolors='darkred', linewidths=1.5)
    ax_map.scatter(model_lon, model_lat, c='dodgerblue', s=150, marker='s', label='Model grid', zorder=5, edgecolors='darkblue', linewidths=1.5)
    
    # Draw line connecting the two points
    ax_map.plot([lon, model_lon], [lat, model_lat], 'k--', linewidth=1.5, alpha=0.7, zorder=4)
    
    ax_map.set_xlabel('Longitude', fontsize=11)
    ax_map.set_ylabel('Latitude', fontsize=11)
    ax_map.set_title(f'Model Domain - Profile Location (dist: {distance_km:.2f} km)', fontsize=13)
    ax_map.legend(loc='upper right', fontsize=10)
    ax_map.grid(True, alpha=0.3)
    
    # ===== Profile plots =====
    # Extract and plot model profiles for each matching day
    for time_idx, day_offset in matching_days:
        T_model, S_model = extract_model_profile(ds_model, time_idx, x_idx, y_idx)
        
        # Color based on offset: darker for exact day, lighter for farther days
        alpha = 0.3 + 0.4 * (1 - abs(day_offset) / DAYS_WINDOW)
        color = 'dodgerblue'
        
        if day_offset == 0:
            # Exact matching day - highlight it
            ax1.plot(-z_model, T_model, color='dodgerblue', alpha=0.8, linewidth=2, 
                    label=f'Model (day {time_idx+1})' if day_offset == 0 else None)
            ax2.plot(-z_model, S_model, color='dodgerblue', alpha=0.8, linewidth=2,
                    label=f'Model (day {time_idx+1})' if day_offset == 0 else None)
        else:
            ax1.plot(-z_model, T_model, color='dodgerblue', alpha=alpha, linewidth=1)
            ax2.plot(-z_model, S_model, color='dodgerblue', alpha=alpha, linewidth=1)
    
    # Plot CTD observation profile - on top
    ax1.plot(depth_ctd, T_ctd, 'r-', linewidth=2.5, label=f'CTD ({obs_date})')
    ax2.plot(depth_ctd, S_ctd, 'r-', linewidth=2.5, label=f'CTD ({obs_date})')
    
    # Temperature plot settings
    ax1.set_xlabel('Depth (m)', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title('Temperature Profile', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Depth increases to the right typically, but inverted for profile view
    
    # Actually, for profiles it's more intuitive to have depth on Y-axis
    # Let's redo with depth on Y-axis
    ax1.clear()
    ax2.clear()
    
    # Re-plot with depth on Y-axis (profile view)
    for time_idx, day_offset in matching_days:
        T_model, S_model = extract_model_profile(ds_model, time_idx, x_idx, y_idx)
        
        alpha = 0.3 + 0.4 * (1 - abs(day_offset) / DAYS_WINDOW)
        
        # z_model is typically negative, so use -z_model for positive depth
        depth_model = -z_model if z_model[0] < 0 else z_model
        
        if day_offset == 0:
            ax1.plot(T_model, depth_model, color='dodgerblue', alpha=0.8, linewidth=2, 
                    label=f'Model (day {time_idx+1})')
            ax2.plot(S_model, depth_model, color='dodgerblue', alpha=0.8, linewidth=2,
                    label=f'Model (day {time_idx+1})')
        else:
            ax1.plot(T_model, depth_model, color='dodgerblue', alpha=alpha, linewidth=1)
            ax2.plot(S_model, depth_model, color='dodgerblue', alpha=alpha, linewidth=1)
    
    # Plot CTD observation profile - on top
    ax1.plot(T_ctd, depth_ctd, color='tomato', linewidth=2.5, label=f'CTD ({obs_date})')
    ax2.plot(S_ctd, depth_ctd, color='tomato', linewidth=2.5, label=f'CTD ({obs_date})')
    
    # Temperature plot settings
    ax1.set_ylabel('Depth (m)', fontsize=12)
    ax1.set_xlabel('Temperature (°C)', fontsize=12)
    ax1.set_title('Temperature Profile', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Depth increases downward
    
    # Salinity plot settings
    ax2.set_ylabel('Depth (m)', fontsize=12)
    ax2.set_xlabel('Salinity (PSU)', fontsize=12)
    ax2.set_title('Salinity Profile', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Depth increases downward
    
    # Main title
    fig.suptitle(f'{profile_id}\nLat: {lat:.4f}°N, Lon: {lon:.4f}°W | DOY: {obs_doy}\n'
                 f'Model: ±{DAYS_WINDOW} days window (semi-transparent lines)', 
                 fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / f'{profile_id}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    generated_count += 1
    print(f"  Saved: {output_path.name}")

print(f"\n{'='*60}")
print(f"Completed!")
print(f"Generated: {generated_count} comparison plots")
print(f"Skipped (distance > {MAX_DISTANCE_KM} km): {skipped_distance}")
print(f"Output directory: {OUTPUT_DIR}")
