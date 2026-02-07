# SAFARI Mooring

Code and notebooks to process SAFARI mooring data, compute air-sea fluxes with COARE 3.6, and generate plots/diagnostics for the project.

**Contents**
1. `src/` Python analysis scripts and utilities
2. `data/` Local outputs (e.g., NetCDF flux products)
3. `img/` Figures

**Key Scripts**
1. `src/SAFARI_realtime_flux.py`
   Computes hourly fluxes using `coare36vn_zrf_et.py` and writes a NetCDF file.
2. `src/coare36vn_zrf_et.py`
   COARE 3.6 bulk flux implementation with optional albedo outputs.
3. `src/coare36_variables_config.py`
   Metadata for COARE outputs (names, units, CF-ish attributes, sign conventions).
4. `src/plot_SAFARI_SST_context.py`
   SST context plotting (currently blocked; see note below).

**Outputs**
`src/SAFARI_realtime_flux.py` writes a NetCDF file to:
`data/SAFARI_fluxes.nc`

**SST Plotting Status**
SST context plotting is **not working right now** because there appears to be an issue reaching the NOAA THREDDS server used by `src/plot_SAFARI_SST_context.py`.

**Usage**
Run scripts from the repo root or `src/` directory as you normally do in your workflow. Most scripts are intended to be executed as stand‑alone analysis steps.

