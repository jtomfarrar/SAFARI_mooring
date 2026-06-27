# Project Context

This file is for durable context that may be useful across sessions, agents, or
long gaps between edits. Use it for project state that is not obvious from the
code alone: data provenance, known blockers, workflow assumptions, script status,
and decisions that would otherwise need to be rediscovered.

Keep this file concise and current. Prefer brief notes with dates or status
labels over detailed implementation explanations that belong in code comments,
docstrings, or the README.

## What To Track Here

- Current purpose and status of scripts, especially analysis scripts whose role
  is not clear from their filename alone.
- Known blockers, missing data, external services, or environment requirements.
- Important workflow assumptions, such as where scripts should be run from and
  which outputs are expected to be regenerated.
- Decisions made during analysis that affect future interpretation of results.

## File And Status Notes

The README gives the broad repository overview. This section should capture
working context that future agents may need before editing or running files.

- `src/SAFARI_realtime_flux.py`: Computes SAFARI mooring air-sea fluxes and
  writes `data/SAFARI_fluxes.nc`.
- `src/coare36vn_zrf_et.py`: COARE 3.6 bulk flux implementation used by the
  flux workflow.
- `src/coare36_variables_config.py`: Metadata and naming configuration for COARE
  output variables.
- `src/plot_SAFARI_ERA5_comparison.py`: Generates SAFARI/ERA5 comparison
  diagnostics in `img/`.
- `src/SAFARI_ERA5_daily_climatology.py`: Generates daily climatology products
  and related figures.
- `src/plot_SAFARI_wirewalker.py`: Loads SAFARI Wirewalker up-profile JSON data,
  generates wirewalker-related plots, and writes
  `data/SAFARI_wirewalker_profiles.nc`.
- `src/plot_SAFARI_ocean_heat_budget.py`: Loads flux, Wirewalker, and L3 met
  NetCDF products, augments Wirewalker surface T/S from the met file, fills the
  upper T/S gap by vertical interpolation, estimates density-threshold mixed
  layer depth, and makes the ocean heat budget overview plus fixed-depth and
  MLD-following temperature-budget diagnostic plots.
- `src/plot_SAFARI_SST_context.py`: SST context plotting script. The README
  notes this is currently blocked by an issue reaching the NOAA THREDDS server.
- `src/download_DUACS.py`: DUACS download helper.
- `data/`: Local NetCDF outputs and intermediate data products.
- `img/`: Generated figures.

## Agent Notes

Before editing Python code, follow `AGENTS.md`, which points to the shared
Python style guidance in `../AGENTS.md`.

## Ocean Heat Budget Development Plan

Build the heat-budget work incrementally in
`src/plot_SAFARI_ocean_heat_budget.py`, keeping computed budget terms in xarray
objects before adding new plots. The current script already creates an
augmented Wirewalker dataset (`ds_ww_aug`) with buoy SST/SSS in the first two
depth rows, vertical T/S interpolation through the upper gap, sigma0-based MLD
using a 0.5 kg/m^3 threshold, and surface heat flux components from
`SAFARI_fluxes.nc`.

### Theory
This section is durable scientific context for future budget extensions. It can
be skipped for routine script maintenance, but read it before adding advection,
revisiting entrainment, changing vertical flux divergence terms, or extending
the work to salt budgets.

The depth $h$ is often chosen to be the mixed layer depth, but the derivation holds for any choice of $h$. Without further approximation, the equation
governing the depth-average temperature between the surface and the depth $h$ is
\begin{eqnarray}
  \frac{\partial T_a}{\partial t} + {\bf u}_a\cdot\nabla T_a + \frac{1}{h}\nabla\cdot\int^0_{-h} {\bf\hat{u}}\hat{T}dz+\frac{\hat{T}_{-h}}{h}(\frac{\partial h}{\partial t}+{\bf
  u}_{-h}\cdot\nabla h +w_{-h})=\frac{Q_o-Q_{-h}}{\rho c_p h} \ \ \  \label{Tbal_eqn}\\
  A \ \ \ \ \ \ \ \ B \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ C \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  D \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ E \ \ \ \ \ \ \
  \nonumber
\end{eqnarray}
where the subscript $a$ indicates the vertical average to depth $h$, e.g.,
\begin{equation}
  T_a=\frac{1}{h}\int^0_{-h}Tdz,
\end{equation}
and the `hats' indicate deviations from the vertical average, so that, e.g., $T(z)=\hat{T}(z) + T_a$.  Equation \ref{Tbal_eqn} states that the rate of change of vertically averaged temperature (term A) is influenced by horizontal advection of the layer-average temperature gradient by the layer-average velocity (term B), vertical entrainment (term D), and the net vertical divergence of vertical heat fluxes over the layer (term E). There is also a contribution to the vertically averaged temperature from a term associated with correlated vertical variations of temperature and velocity (term C), which has the form of the divergence of a Reynolds correlation (or `eddy flux').  (Term C is not an `eddy flux' in the conventional meaning of the term; that is, it is not associated with mesoscale or turbulent fluctuations, per se.)

### Source MATLAB Context

The user provided `BoB_TS_balance_buoy_v1.m` as a model for the method. The key
temperature-budget pieces in that script are:

- Layer-average temperature `Tbar` from the surface to depth `H`.
- Tendency `Tt = diff(Tbar2) / 3600`, after interpolating `Tbar` to hourly flux
  time in the MATLAB workflow.
- Surface heat-flux tendency `Qterm = QN / rho / cp / H2`.
- Penetrating-solar correction using Paulson-Simpson/Jerlov IA:
  `I_pen_ratio = R_PS * exp(-H2 / lambda1) + (1 - R_PS) * exp(-H2 / lambda2)`,
  with `R_PS = 0.62`, `lambda1 = 0.6 m`, and `lambda2 = 20 m`.
- Penetrating-solar tendency `Qpen = -Qs * I_pen_ratio / rho / cp / H2`.
- Vertical-diffusion proxy `QminusH = Tz_minusH2 * 5e-5 / H2`, deferred until
  the first deterministic A/E comparison works.
- Entrainment-like term `entr = That_minusH2 * diff(H2) / 3600 / H2`.

MATLAB dependencies that are not essential for the Python heat-budget workflow:
`smooth_2d`, `ecolorbar`, `export_fig`, `rgb`, `packrows`, `errorbar_tick`,
`badvalreplace`, `gravity`, and `smooth_1d`. Do not port MATLAB smoothing unless
explicitly requested; the Python workflow should use a transparent xarray-based
rolling mean.

### Implemented Python Budgets

The script includes a fixed-depth heat budget and an MLD-following heat budget.
The fixed-depth budget uses `h = 150` m as a configurable constant. The
MLD-following budget uses `H2 = mixed_layer_depth`. Both follow the MATLAB
workflow by interpolating the augmented Wirewalker data to the hourly flux time
grid. Duplicate Wirewalker timestamps are averaged before interpolation because
xarray requires unique time values.

Use these constants unless later MATLAB-port details require changes:

- `rho0 = 1025` kg/m^3
- `cp = 3990` J/kg/K
- `seconds_per_day = 86400`
- `R_PS = 0.62`
- `lambda1 = 0.6`
- `lambda2 = 20`

The fixed-depth budget is stored in `ds_budget`; the MLD-following budget is
stored in `ds_budget_mld`. Both use `coords={'time': ds_flux.time}`. Store
MATLAB-style variable names in these datasets and keep tendencies internally in
degC/s; plot them in degC/day.

Implementation steps:

- Compute `Tbar` by interpolating the flux-time Wirewalker temperature to
  include the layer depth (`h` or `H2`), selecting depths from the surface to
  that depth, and trapezoid-integrating over depth.
- Compute term A, `Tt`, using centered finite differences with actual elapsed
  seconds between flux timestamps.
- Define `QN = sensible_heat_flux + latent_heat_flux + net_solar_radiation +
  net_longwave_radiation`.
- Define `Qs = net_solar_radiation`.
- Define `Qterm = QN / rho0 / cp / h` for fixed-depth, and the same expression
  with `H2` for the MLD-following budget.
- Define `I_pen_ratio = R_PS * exp(-h / lambda1) + (1 - R_PS) *
  exp(-h / lambda2)`, with `h` replaced by `H2` for the MLD-following budget.
- Define `Qpen = -Qs * I_pen_ratio / rho0 / cp / h`, with `h` replaced by
  `H2` for the MLD-following budget.
- Define `Qterm_plus_Qpen = Qterm + Qpen`.
- Define `Tz_h` as the vertical temperature gradient at `h`.
- Define `QminusH = Tz_h * 5e-5 / h`, following the MATLAB
  vertical-diffusion proxy.
- Define `Qsum = Qterm_plus_Qpen + QminusH`.
- For the MLD-following budget, compute `That_minusH = T(H2) - Tbar`,
  `dh_dt`, and `entrainment_tendency = That_minusH * dh_dt / H2`. Compute
  `dh_dt` with `np.gradient(H2, time_seconds)` so the derivative stays on the
  flux-time grid and NaN influence remains local to the finite-difference
  stencil.
- For the MLD-following budget, define
  `Qsum = Qterm_plus_Qpen + QminusH + entrainment_tendency`.

The budget diagnostic plots compare:

- `Tt`
- `Qterm`
- `Qterm_plus_Qpen`
- `QminusH`
- `entrainment_tendency` for the MLD-following budget
- `Qsum`

Plot these in degC/day with a zero line. Add smoothing only for display. Specify
the smoothing window as a time duration, not as the primary user-facing number
of samples. The current display smoother is set to 14 days. Current xarray
`rolling()` takes integer sample windows rather than time-duration strings, so
convert the duration to a sample count only after checking that the budget time
grid is regular. Do not convert to pandas for this smoothing.

### Error Bars

MLD-following temperature-budget error bars are implemented as a first-pass
uncertainty estimate based on the MATLAB error formulas. They use the centered
xarray rolling display smoother and estimate uncertainties for `Tt`, `Qterm`,
`entrainment_tendency`, `QminusH`, and `Qsum`. The plotted errors are sparse
with an 11-day cadence.

Important error constants currently use the MATLAB values:
`delHonH = 0.08`, `delTinstr = 0.01`, `delQ = 8 W/m^2`, `delta_t = 3600 s`,
and `del_FonF = 1`. These should be revisited for SAFARI sampling and
instrumentation.

### Remaining Extensions

- Add fixed-depth temperature error bars if needed.
- Make the display smoother time-duration based in configuration, with an
  internal sample-count conversion after validating regular time spacing.
- Consider adding a salt budget later, but do not mix that into the first heat
  budget implementation.

### Current Assumptions

- Use hourly flux times as the first budget time grid.
- Use `net_solar_radiation` as MATLAB `Qs`.
- Use the existing overview net heat-flux convention as MATLAB `QN`.
- For fixed-depth `h`, entrainment from `dh/dt` is zero and deferred.
