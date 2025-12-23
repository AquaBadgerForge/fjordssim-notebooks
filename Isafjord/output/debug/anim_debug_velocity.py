import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
import numpy as np

# Load the dataset and select the surface level
ds_all = xr.open_dataset('C:/Users/fisa/Documents/Isafjord/output/snapshots_ocean.nc').isel(z_aac=27)
print(ds_all.keys())

# Mask zeros with NaNs for velocity variables
for var in ['u', 'v']:
    if var in ds_all:
        ds_all[var] = ds_all[var].where(ds_all[var] != 0, np.nan)
        valid_data = ds_all[var].values[~np.isnan(ds_all[var].values)]
        if len(valid_data) > 0:
            print(f"{var}: min={valid_data.min():.6f}, max={valid_data.max():.6f}")
        else:
            print(f"{var}: all values masked")

# Load required vars
time = ds_all.time.values
nt = len(time)

# Coordinates
xC, yC = ds_all['λ_caa'].values, ds_all['φ_aca'].values  # Cell centers
xF, yF = ds_all['λ_faa'].values, ds_all['φ_afa'].values  # Cell faces

# Color scale limits
vminmax = {}
vminmax['u'] = (-0.3, 0.3)
vminmax['v'] = (-0.3, 0.9)

# Setup figure with 2 subplots for u and v
fig, axs = plt.subplots(1, 2, figsize=(16, 8), facecolor='black')
fig.patch.set_facecolor('black')
plots = {}
colorbars = {}

# Store zoom information
zoom_info = {'current_var': 'u', 'frame_count': 0}


def find_extrema_location(var_data, xcoords, ycoords):
    """Find location of maximum absolute value"""
    abs_data = np.abs(var_data)
    # Mask NaN values
    masked_data = np.ma.masked_invalid(abs_data)
    if masked_data.count() == 0:
        return None, None, 0
    
    max_idx = np.unravel_index(np.ma.argmax(masked_data), masked_data.shape)
    max_val = var_data[max_idx]
    
    # Get coordinates
    if len(xcoords.shape) == 1:
        x_loc = xcoords[max_idx[1]] if max_idx[1] < len(xcoords) else xcoords[-1]
    else:
        x_loc = xcoords[max_idx]
    
    if len(ycoords.shape) == 1:
        y_loc = ycoords[max_idx[0]] if max_idx[0] < len(ycoords) else ycoords[-1]
    else:
        y_loc = ycoords[max_idx]
    
    return x_loc, y_loc, max_val


def init():
    ds = ds_all.isel(time=0)
    variables = ['u', 'v']
    titles = ['U Velocity (m/s)', 'V Velocity (m/s)']
    cmaps = ['RdBu_r', 'RdBu_r']
    
    for idx, (var, title, cmap) in enumerate(zip(variables, titles, cmaps)):
        ax = axs[idx]
        v = ds[var].values
        
        if var == 'u':
            xFm, yCm = np.meshgrid(xF, yC)
            plots[var] = ax.pcolormesh(xFm, yCm, v, cmap=cmap,
                                vmin=vminmax[var][0], vmax=vminmax[var][1])
        elif var == 'v':
            xCm, yFm = np.meshgrid(xC, yF)
            plots[var] = ax.pcolormesh(xCm, yFm, v, cmap=cmap,
                                vmin=vminmax[var][0], vmax=vminmax[var][1])
        
        ax.set_title(title, color='white')
        ax.set_facecolor('black')
        ax.grid(True, alpha=0.3, color='gray')
        ax.set_xlabel('Longitude', color='white')
        ax.set_ylabel('Latitude', color='white')
        ax.tick_params(colors='white')
        
        colorbars[var] = fig.colorbar(plots[var], ax=ax, label=f'{var} (m/s)')
        colorbars[var].ax.yaxis.set_tick_params(color='white')
        colorbars[var].outline.set_edgecolor('white')
        plt.setp(plt.getp(colorbars[var].ax.axes, 'yticklabels'), color='white')
        colorbars[var].set_label(f'{var} (m/s)', color='white')
    
    return plots.values()


def update(i):
    ds = ds_all.isel(time=i)
    
    # Update data
    for var in plots:
        v = ds[var].values
        plots[var].set_array(v.ravel())
    
    # Determine which variable to zoom to (alternate every 10 frames)
    zoom_info['frame_count'] += 1
    if zoom_info['frame_count'] % 10 == 0:
        zoom_info['current_var'] = 'v' if zoom_info['current_var'] == 'u' else 'u'
    
    # Find extrema and zoom
    for idx, var in enumerate(['u', 'v']):
        ax = axs[idx]
        v = ds[var].values
        
        if var == 'u':
            x_loc, y_loc, max_val = find_extrema_location(v, xF, yC)
        else:
            x_loc, y_loc, max_val = find_extrema_location(v, xC, yF)
        
        if x_loc is not None and y_loc is not None:
            # Zoom window size (smaller = more zoom)
            zoom_size = 0.15  # degrees
            
            # If this is the variable we're focusing on, zoom in more
            if var == zoom_info['current_var']:
                zoom_size = 0.08
                ax.set_title(f'{var.upper()} Velocity (m/s) - Max: {max_val:.3f} m/s ⭐', 
                           fontweight='bold', fontsize=14, color='white')
            else:
                ax.set_title(f'{var.upper()} Velocity (m/s) - Max: {max_val:.3f} m/s', color='white')
            
            # Set zoom limits
            ax.set_xlim(x_loc - zoom_size, x_loc + zoom_size)
            ax.set_ylim(y_loc - zoom_size, y_loc + zoom_size)
            
            # Mark extrema location
            ax.plot(x_loc, y_loc, 'r*', markersize=15, markeredgecolor='white', 
                   markeredgewidth=1.5, zorder=10)
    
    # Convert time to hours/days for display
    time_delta = pd.Timedelta(time[i])
    hours = time_delta.total_seconds() / 3600
    days = hours / 24
    fig.suptitle(f"Time: {days:.3f} days ({hours:.1f} hours) - Focusing on {zoom_info['current_var'].upper()}", 
                fontsize=16, fontweight='bold', color='white')
    
    print(f"Frame {i}/{nt} updated - Zoom focus: {zoom_info['current_var'].upper()}")
    return plots.values()


# Animate
anim = FuncAnimation(fig, update, init_func=init, frames=nt, blit=False)

# Save as MP4
writer = FFMpegWriter(fps=2, metadata=dict(artist='Python'), bitrate=5000)
anim.save("velocity_zoom_animation.mp4", writer=writer)
print(f"Animation saved! Total frames: {nt}")

plt.close()
