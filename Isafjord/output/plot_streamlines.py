import xarray as xr
import matplotlib.pyplot as plt
# import contextily as ctx
import numpy as np
import colormaps as cmaps
import os

plt.style.use('dark_background')

# === User Parameters ===
window_size = 90  # Number of timesteps to average
output_dir = "visual/exp_3dec_1year/streamlines"  # Output directory
zlayer = 25  # Surface layer index 27
iz1, iz2 = 14, 27  # Slicing for depth levels
os.makedirs(output_dir, exist_ok=True)

vmin_speed, vmax_speed = 0, 0.15  # Speed range (m/s)
cmap_speed = cmaps.WhiteBlueGreenYellowRed  # Speed colormap
density = 2  # Streamline density (higher = more lines)

# === Load Data ===
ds_all = xr.open_dataset('C:/Users/fisa/Documents/Isafjord/output/3dec_T5_S34.5/snapshots_ocean_3dec.nc', decode_timedelta=True).isel(z_aac=slice(0, 28))
print("Dataset loaded successfully")

# Coordinates
# xC, yC = ds_all['λ_caa'], ds_all['φ_aca']  # Cell centers
# xF, yF = ds_all['λ_faa'], ds_all['φ_afa']  # Cell faces

# Grid spacing (meters)
dx = 181.12762
dy = 222.38985

# Construct coordinates
nx = 256
ny = 210
x_coords = np.arange(nx) * dx
y_coords = np.arange(ny) * dy

# Create meshgrids
xCm, yCm = np.meshgrid(x_coords, y_coords)
# xFm, yFm = np.meshgrid(xF, yF)

# Get time information
time = ds_all.time.values
nt = len(time)

# Calculate number of windows
n_windows = nt // window_size
print(f"Total timesteps: {nt}")
print(f"Window size: {window_size}")
print(f"Number of windows: {n_windows}")

# === Process each window ===
for window_idx in range(n_windows):
    start_idx = window_idx * window_size
    end_idx = start_idx + window_size
    
    print(f"\nProcessing window {window_idx + 1}/{n_windows} (timesteps {start_idx}-{end_idx})")
    
    # Average velocities over the window at surface level
    # u is on (φ_aca, λ_faa) - face grid in x
    # v is on (φ_afa, λ_caa) - face grid in y
    u_avg = ds_all['u'].isel(time=slice(start_idx, end_idx), z_aac=slice(iz1, iz2)).mean(dim=['time', 'z_aac']).values
    v_avg = ds_all['v'].isel(time=slice(start_idx, end_idx), z_aac=slice(iz1, iz2)).mean(dim=['time', 'z_aac']).values
    
    # Mask zero values
    u_avg = np.where(u_avg == 0, np.nan, u_avg)
    v_avg = np.where(v_avg == 0, np.nan, v_avg)
    
    # Interpolate u and v to cell centers (φ_aca, λ_caa)
    # u: average adjacent x-face values to get cell center
    u_center = 0.5 * (u_avg[:, :-1] + u_avg[:, 1:])
    # v: average adjacent y-face values to get cell center
    v_center = 0.5 * (v_avg[:-1, :] + v_avg[1:, :])
    
    # Both should now be on the same grid
    min_shape = (min(u_center.shape[0], v_center.shape[0]), min(u_center.shape[1], v_center.shape[1]))
    u_trimmed = u_center[:min_shape[0], :min_shape[1]]
    v_trimmed = v_center[:min_shape[0], :min_shape[1]]
    speed = np.sqrt(u_trimmed**2 + v_trimmed**2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot speed as background
    xCm_trim = xCm[:min_shape[0], :min_shape[1]]
    yCm_trim = yCm[:min_shape[0], :min_shape[1]]
    
    pcm = ax.pcolormesh(xCm_trim, yCm_trim, speed, 
                        cmap=cmap_speed, 
                        vmin=vmin_speed, vmax=vmax_speed, 
                        shading='auto', alpha=0.7)
    
    # Plot streamlines - use cell center coordinates
    # u_trimmed and v_trimmed are (y, x) = (210, 256)
    
    strm = ax.streamplot(x_coords, y_coords, u_trimmed, v_trimmed,
                         color='white', linewidth=1.5, density=density,
                         arrowsize=1.2, arrowstyle='->')
    
    # Add basemap
    # ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.DarkMatter, attribution_size=6)
    
    # Calculate time range in days
    import pandas as pd
    start_time = pd.Timedelta(time[start_idx])
    end_time = pd.Timedelta(time[end_idx - 1])
    start_days = start_time.total_seconds() / 86400
    end_days = end_time.total_seconds() / 86400
    
    # Title and labels
    depth1, depth2 = ds_all['z_aac'].isel(z_aac=[iz1, iz2]).values

    ax.set_title(f'Streamlines {depth2:.1f} - {depth1:.1f} m (Averaged)\nDays {start_days:.1f} - {end_days:.1f}', 
                 fontsize=14, color='white')
    ax.set_xlabel('Longitude', color='white')
    ax.set_ylabel('Latitude', color='white')
    ax.tick_params(colors='white')
    
    # Colorbar
    cbar = plt.colorbar(pcm, ax=ax, label='Speed (m/s)', shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    
    # Save figure
    fname = f"{output_dir}/stream{depth2:.1f}-{depth1:.1f}m_days_{start_days:.0f}-{end_days:.0f}.png"
    plt.savefig(fname, 
                dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved: {fname}")

print("\nAll streamline plots completed!")
