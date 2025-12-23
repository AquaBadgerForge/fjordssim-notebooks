"""
Compare CTD profiles from observations with model output.
Groups nearby CTD profiles (within 400m) as the same station and plots
all observations together with model data.
"""

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import contextily as ctx
from collections import defaultdict

# ============================================================================
# Configuration
# ============================================================================

# Paths
MODEL_PATH = 'C:/Users/fisa/Documents/Isafjord/output/3dec_T5_S34.5/snapshots_ocean_3dec.nc'
PROFILE_DB_PATH = Path('C:/Users/fisa/Documents/Isafjord/validation/profile_database')
OUTPUT_DIR = Path('C:/Users/fisa/Documents/fjordssim-notebooks/Isafjord/output/visual/exp_3dec_1year/joined_ctds_vs_model')

# Days window around observation for model comparison
DAYS_WINDOW = 7

# Maximum distance (km) between observation and model grid point
MAX_DISTANCE_KM = 10

# Maximum distance (km) to consider profiles as same station
STATION_GROUPING_DISTANCE_KM = 0.4  # 400 meters

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
model_doy = np.arange(1, nt + 1)

# ============================================================================
# Load profile database
# ============================================================================

print("\nLoading profile database...")
index_df = pd.read_csv(PROFILE_DB_PATH / 'index.csv')
index_df['datetime'] = pd.to_datetime(index_df['datetime'])
index_df['doy'] = index_df['datetime'].dt.dayofyear
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
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

def find_nearest_grid_point(lon, lat, xC, yC):
    """Find nearest grid indices for a given lon/lat and return distance."""
    x_idx = np.argmin(np.abs(xC - lon))
    y_idx = np.argmin(np.abs(yC - lat))
    distance_km = haversine_distance(lon, lat, xC[x_idx], yC[y_idx])
    return x_idx, y_idx, distance_km

def extract_model_profile(ds, time_idx, x_idx, y_idx):
    """Extract T and S profiles from model at given location and time."""
    T_profile = ds['T'].isel(time=time_idx).values[:, y_idx, x_idx]
    S_profile = ds['S'].isel(time=time_idx).values[:, y_idx, x_idx]
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
    """
    matching_indices = []
    for day_offset in range(-days_window, days_window + 1):
        target_doy = obs_doy + day_offset
        if target_doy < 1:
            target_doy += 365
        elif target_doy > 365:
            target_doy -= 365
        if 1 <= target_doy <= len(model_doy):
            matching_indices.append((target_doy - 1, day_offset))
    return matching_indices

def group_profiles_by_location(df, max_distance_km):
    """
    Group profiles that are within max_distance_km of each other.
    Returns a list of groups, where each group is a list of row indices.
    """
    n = len(df)
    visited = [False] * n
    groups = []
    
    for i in range(n):
        if visited[i]:
            continue
        
        # Start a new group
        group = [i]
        visited[i] = True
        
        lat_i = df.iloc[i]['latitude']
        lon_i = df.iloc[i]['longitude']
        
        # Find all profiles within distance
        for j in range(i + 1, n):
            if visited[j]:
                continue
            
            lat_j = df.iloc[j]['latitude']
            lon_j = df.iloc[j]['longitude']
            
            dist = haversine_distance(lon_i, lat_i, lon_j, lat_j)
            if dist <= max_distance_km:
                group.append(j)
                visited[j] = True
        
        groups.append(group)
    
    return groups

# ============================================================================
# Group profiles by location
# ============================================================================

print(f"\nGrouping profiles within {STATION_GROUPING_DISTANCE_KM*1000:.0f}m of each other...")
station_groups = group_profiles_by_location(index_df, STATION_GROUPING_DISTANCE_KM)
print(f"Found {len(station_groups)} station groups from {len(index_df)} profiles")

# Count groups with multiple profiles
multi_profile_groups = sum(1 for g in station_groups if len(g) > 1)
print(f"Groups with multiple profiles: {multi_profile_groups}")

# ============================================================================
# Create output directory
# ============================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\nOutput directory: {OUTPUT_DIR}")

# ============================================================================
# Generate comparison plots for each station group
# ============================================================================

print("\nGenerating comparison plots...")

# Color palette for different CTD observations
ctd_colors = plt.cm.Set1(np.linspace(0, 1, 9))  # Up to 9 distinct colors

skipped_distance = 0
generated_count = 0

for group_idx, group in enumerate(station_groups):
    # Get representative location (first profile in group)
    first_row = index_df.iloc[group[0]]
    lat = first_row['latitude']
    lon = first_row['longitude']
    
    # Create station name from first profile
    station_name = f"station_{group_idx+1:04d}"
    
    print(f"\nProcessing group {group_idx+1}/{len(station_groups)}: {station_name} ({len(group)} profiles)")
    print(f"  Location: ({lat:.4f}, {lon:.4f})")
    
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
    
    # Get model grid point coordinates
    model_lon = xC[x_idx]
    model_lat = yC[y_idx]
    
    # Collect all CTD profiles and their metadata
    ctd_data = []
    all_doys = set()
    
    for profile_idx in group:
        row = index_df.iloc[profile_idx]
        ctd_filepath = PROFILE_DB_PATH / row['filepath']
        try:
            depth_ctd, T_ctd, S_ctd = load_ctd_profile(ctd_filepath)
            ctd_data.append({
                'profile_id': row['profile_id'],
                'cruise': row['cruise'],
                'date': row['date'],
                'doy': row['doy'],
                'depth': depth_ctd,
                'T': T_ctd,
                'S': S_ctd,
                'lat': row['latitude'],
                'lon': row['longitude']
            })
            all_doys.add(row['doy'])
        except Exception as e:
            print(f"  ERROR loading CTD profile {row['profile_id']}: {e}")
            continue
    
    if not ctd_data:
        print(f"  WARNING: No valid CTD profiles loaded, skipping")
        continue
    
    # Get all matching model days across all observations
    all_matching_days = set()
    for doy in all_doys:
        matching = get_matching_model_days(doy, model_doy, DAYS_WINDOW)
        for m in matching:
            all_matching_days.add(m)
    
    all_matching_days = sorted(all_matching_days, key=lambda x: x[0])
    
    if not all_matching_days:
        print(f"  WARNING: No matching model days found, skipping")
        continue
    
    # Create figure with map on top row, profiles on bottom row
    fig = plt.figure(figsize=(12, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 2], hspace=0.25, wspace=0.25)
    
    ax_map = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    
    # ===== Map plot (full model domain) =====
    ax_map.set_xlim(xC.min(), xC.max())
    ax_map.set_ylim(yC.min(), yC.max())
    
    try:
        ctx.add_basemap(ax_map, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, attribution_size=5)
    except Exception as e:
        ax_map.set_facecolor('lightblue')
        print(f"  Warning: Could not add basemap: {e}")
    
    # Plot all observation points in the group
    for i, ctd in enumerate(ctd_data):
        color = ctd_colors[i % len(ctd_colors)]
        ax_map.scatter(ctd['lon'], ctd['lat'], c=[color], s=100, marker='o', 
                      zorder=5, edgecolors='black', linewidths=1, alpha=0.8)
    
    # Plot model grid point
    ax_map.scatter(model_lon, model_lat, c='dodgerblue', s=150, marker='s', 
                  label='Model grid', zorder=6, edgecolors='darkblue', linewidths=1.5, alpha=0.6)
    
    # Draw line from centroid to model point
    ax_map.plot([lon, model_lon], [lat, model_lat], 'k--', linewidth=1.5, alpha=0.7, zorder=4)
    
    ax_map.set_xlabel('Longitude', fontsize=11)
    ax_map.set_ylabel('Latitude', fontsize=11)
    ax_map.set_title(f'Model Domain - {len(ctd_data)} CTD profiles at station (dist: {distance_km:.2f} km)', fontsize=13)
    ax_map.grid(True, alpha=0.3)
    
    # ===== Profile plots =====
    depth_model = -z_model if z_model[0] < 0 else z_model
    
    # Plot CTD observation profiles with model profiles matching same color per date
    for i, ctd in enumerate(ctd_data):
        color = ctd_colors[i % len(ctd_colors)]
        obs_doy = ctd['doy']
        
        # Get model profiles for this specific observation date (±DAYS_WINDOW)
        matching_days = get_matching_model_days(obs_doy, model_doy, DAYS_WINDOW)
        
        # Plot model profiles (thin, semi-transparent, same color as observation)
        for time_idx, day_offset in matching_days:
            T_model, S_model = extract_model_profile(ds_model, time_idx, x_idx, y_idx)
            alpha = 0.15 + 0.25 * (1 - abs(day_offset) / DAYS_WINDOW)
            ax1.plot(T_model, depth_model, color=color, alpha=alpha, linewidth=1)
            ax2.plot(S_model, depth_model, color=color, alpha=alpha, linewidth=1)
        
        # Plot observation profile (thick line)
        label = f"{ctd['date']} - {ctd['cruise']}"
        ax1.plot(ctd['T'], ctd['depth'], color=color, linewidth=2.5, alpha=1.0, label=label)
        ax2.plot(ctd['S'], ctd['depth'], color=color, linewidth=2.5, alpha=1.0, label=label)
    
    # Add legend entry for model (use first color as example)
    ax1.plot([], [], color='gray', alpha=0.4, linewidth=1, linestyle='-', label=f'Model (±{DAYS_WINDOW} days)')
    ax2.plot([], [], color='gray', alpha=0.4, linewidth=1, linestyle='-', label=f'Model (±{DAYS_WINDOW} days)')
    
    # Temperature plot settings
    ax1.set_ylabel('Depth (m)', fontsize=12)
    ax1.set_xlabel('Temperature (°C)', fontsize=12)
    ax1.set_title('Temperature Profile', fontsize=14)
    ax1.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # Salinity plot settings
    ax2.set_ylabel('Depth (m)', fontsize=12)
    ax2.set_xlabel('Salinity (PSU)', fontsize=12)
    ax2.set_title('Salinity Profile', fontsize=14)
    ax2.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # Main title
    profile_ids = ', '.join([c['profile_id'] for c in ctd_data[:3]])
    if len(ctd_data) > 3:
        profile_ids += f', ... (+{len(ctd_data)-3} more)'
    
    fig.suptitle(f'{station_name}: {len(ctd_data)} profiles\n'
                 f'Lat: {lat:.4f}°N, Lon: {lon:.4f}°W\n'
                 f'Model: ±{DAYS_WINDOW} days window', 
                 fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / f'{station_name}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    generated_count += 1
    print(f"  Saved: {output_path.name}")

print(f"\n{'='*60}")
print(f"Completed!")
print(f"Generated: {generated_count} comparison plots")
print(f"Skipped (distance > {MAX_DISTANCE_KM} km): {skipped_distance}")
print(f"Output directory: {OUTPUT_DIR}")
