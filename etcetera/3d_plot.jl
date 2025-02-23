using Oceananigans
using JLD2
using GLMakie
using Oceananigans.Units

# Load the field data from JLD2
folder = joinpath(homedir(), "FjordsSim_results", "oslofjord")
filename = joinpath(folder, "snapshots")
T_series = FieldTimeSeries("$filename.jld2", "T")  # Read temperature data

# Extract grid information from the first field snapshot
Nx, Ny, Nz = size(interior(T_series[1]))
x_range = (0, Nx)
y_range = (0, Ny)
z_range = (-Nz, 0)  # Assuming negative depth

# Time information
times = T_series.times
Nt = length(times)

# Define global min/max for consistent color scale
T_min, T_max = minimum(T_series), maximum(T_series)

# Create a figure
fig = Figure(size=(1000, 800))
ax = Axis3(fig[1, 1], aspect = (1, 3, 1), elevation = 0.2 * pi, azimuth = 1.4 * pi,
           title="Temperature Evolution", xlabel="X", ylabel="Y", zlabel="Depth")

# Observable for 3D temperature field
T_field = Observable(interior(T_series[1]))  # Initialize with first snapshot

# Create a volume plot with fixed color range
vol = volume!(ax, x_range, y_range, z_range, T_field; 
              colormap=Reverse(:RdYlBu), colorrange=(T_min, T_max), transparency=false)

# Add colorbar with fixed range
Colorbar(fig[1, 2], vol, label="T (°C)")

# Animate and record the simulation
record(fig, "temperature_3d.mp4", 1:Nt; framerate=20) do i
    T_field[] = interior(T_series[i])  # Update observable with new timestep data
end