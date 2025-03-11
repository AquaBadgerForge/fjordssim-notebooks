# FjordsSim data preparation

Python and Julia scripts / notebooks for FjordsSim data preparation.

## Installation

Python:

1. Install conda <https://conda-forge.org/download/>.
2. Create a conda environment `conda create --name fjordssim python=3.12`
3. Activate the envoronment `conda activate fjordssim`
4. Navigate to the fjordssim-notebooks directory
5. Install the dependencies `pip install -e .`
6. For some cases xesmf is required, see <https://xesmf.readthedocs.io/en/stable/installation.html>;
   install with `conda install -c conda-forge xesmf`.

Julia:

1. Install julia <https://julialang.org/downloads/>.
2. Run Julia REPL from the directory with `Project.toml` and activate the FjordsSim environment `julia --project=.`.
3. Enter the Pkg REPL by pressing `]` from Julia REPL.
4. Type `instantiate` to 'resolve' a `Manifest.toml` from a `Project.toml` to install and precompile dependency packages.

To run a FjordsSim simulation 2 files are required:

- A bathymetry netcdf file.
It should contain a 2d array variable "h" with depths (they should be negative),
Two 1d arrays with "lat" and "lon" corresponding to depths in variable "h",
a 1d array "z_faces" with the desired layer depths (also negative values).

- A forcing netcdf file.
This file contains the information about the forcing fields.
To 'force' any variable, one need to define two 4d arrays called, for example, "T" and "T_lambda".
"T" is a oceananigans name for temperature, lets use it further as example;
it is possible to provide forcing for any variable defined in an oceananigans simulation.
Spatial dimensions should have the shape of the corresponding [stagerred grid](https://clima.github.io/OceananigansDocumentation/stable/fields/#Staggered-grids-and-field-locations).
The forth dimension is time in seconds from the start of a simulation, in Python one can use a datetime format.
"T_lambda" defines a type of forcing to be used in a simulation.
"T" value should correspond to "T_lambda".
If 0 < "T_lambda" < 1, [relaxation](https://clima.github.io/OceananigansDocumentation/stable/model_setup/forcing_functions/#Relaxation) is used.
If "T_lambda"
