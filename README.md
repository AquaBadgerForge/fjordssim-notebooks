# FjordsSim data preparation

Python and Julia scripts / notebooks for FjordsSim data preparation.

## Installation

Python:

1. Install conda <https://conda-forge.org/download/>.
2. Create a conda environment `conda create --name fjordssim python=3.12`
3. Activate the envoronment `conda activate fjordssim`
4. Install the dependencies `pip install -e .`

Julia:

1. Install julia <https://julialang.org/downloads/>.
2. Run Julia REPL from the directory with `Project.toml` and activate the FjordsSim environment `julia --project=.`.
3. Enter the Pkg REPL by pressing `]` from Julia REPL.
4. Type `instantiate` to 'resolve' a `Manifest.toml` from a `Project.toml` to install and precompile dependency packages.
