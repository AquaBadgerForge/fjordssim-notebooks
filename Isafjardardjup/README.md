# Ísafjarðardjúp (Iceland) Data Preparation

Notebooks and data files for preparing FjordsSim simulation inputs for Ísafjarðardjúp in northwestern Iceland.

## Main Features

### 1. Bathymetry Processing (`Isf_bathymetry.ipynb`)
- Processes bathymetry data for Ísafjarðardjúp
- Creates topography NetCDF files with proper grid structure
- Output files:
  - `Isafjardardjup_topo.nc` - Full resolution topography
  - `Isf_topo299x320.jld2` - Downsampled grid (299×320) in Julia format

### 2. Boundary Conditions (`Isf_BRY.ipynb`)
**Main notebook for creating ocean boundary and river forcing data**

#### Ocean Forcing
- Reads CMEMS (Copernicus Marine Environment Monitoring Service) reanalysis data
- File: `P1D-m_1745438014042.nc` - Daily ocean data (temperature, salinity, currents)
- Interpolates CMEMS data onto model grid
- Creates relaxation boundary conditions for:
  - Temperature (T)
  - Salinity (S)
  - Horizontal velocities (u, v)
- Applies Gaussian smoothing to reduce artificial gradients
- Uses 10-day relaxation timescale for ocean boundaries

#### River Forcing
**Four major rivers are included:**

| River | Station | Coordinates (WGS84) | Mean Discharge |
|-------|---------|---------------------|----------------|
| Þverá | V38 | 65.9067°N, 22.3433°W | ~2.5 m³/s |
| Hvalá | V198 | 66.0677°N, 21.7307°W | ~13 m³/s |
| Dynjandisá | V449 | 65.7372°N, 23.2118°W | ~6 m³/s |
| Vatnsdalsá | V204 | 65.5852°N, 23.1344°W | ~12 m³/s |

**River discharge data** (`rivers_Q.xlsx`):
- Source: **Icelandic Meteorological Office (IMO / Veðurstofa Íslands)**
- Contains daily discharge measurements (Q24M in m³/s)
- Time series: 2014-2024 (10 years)
- Each river has a separate sheet in the Excel file

**River implementation:**
- Geographic coordinates automatically mapped to nearest valid grid points
- Find closest non-NaN bathymetry cells
- River salinity set to 0 (freshwater)
- Relaxation coefficients (lambda) calculated from:
  - λ = Q/V (discharge / cell volume)
  - Lambda values correspond to actual river discharge rates

#### Vertical Grid
- 29 vertical layers with variable thickness
- Surface resolution: 1m layers near surface
- Deep resolution: up to 20m layers at depth
- Depth range: 0 to 190m

#### Output Files
- `Isf_bry_299x320_3d.nc` - Full 3D boundary conditions
- `Isf_bry_299x320.nc` - Alternative boundary configuration
- `Isf_bry_299x320_rivers.nc` - River-specific forcing

### 3. MATLAB to NetCDF Conversion (`mat2nc.ipynb`)
- Converts hypsographic data from MATLAB format
- Source: `Isafjardardjup_hypsography/Isafjardardjup_hypso_tmp_v20190117.mat`

## Grid Configuration

- **Horizontal dimensions**: 299 × 320 grid cells
- **Spatial extent**:
  - Longitude: 22.3°W to 23.5°W
  - Latitude: 65.76°N to 66.40°N
- **Vertical levels**: 29 layers (z-coordinate)
- **Temporal resolution**: Daily (365 days)

## Data Sources

1. **Ocean data**: CMEMS Global Ocean Physics Reanalysis
   - Product: GLOBAL_MULTIYEAR_PHY_001_030
   - Variables: Temperature, salinity, velocities, sea level
   
2. **River discharge**: Icelandic Meteorological Office
   - Daily measurements from gauging stations
   - Quality flags included (Good, Fair, Estimated, Missing)

3. **Bathymetry**: Hypsographic data compilation
   - Regional bathymetry surveys
   - Processed for model grid alignment

## Python Dependencies

Required packages (see main README for installation):
- `xarray` - NetCDF/data handling
- `numpy` - Numerical operations
- `pandas` - Excel file reading, time series
- `matplotlib` - Visualization
- `scipy` - Interpolation and spatial algorithms
- `hvplot` - Interactive plotting
- `netCDF4` - NetCDF file I/O

## Workflow

1. **Prepare bathymetry**: Run `Isf_bathymetry.ipynb`
2. **Create forcing**: Run `Isf_BRY.ipynb`
   - Reads CMEMS ocean data
   - Loads river discharge from `rivers_Q.xlsx`
   - Interpolates to model grid
   - Calculates physics-based relaxation coefficients
   - Outputs NetCDF files for FjordsSim

## Notes

- River positions are automatically interpolated to valid (non-NaN) grid cells
- Ocean boundaries use standard 10-day relaxation
