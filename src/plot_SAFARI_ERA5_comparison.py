# %% SAFARI_mooring - Compare SAFARI L3 met and height-adjusted data with ERA5
# Created: 2026-03-28

# %%
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import sys
from pathlib import Path

# %%
# Set working directory
home_dir = Path.home()
os.chdir(home_dir / 'Python/SAFARI_mooring/src')

# %%
%matplotlib ipympl
plt.rcParams['figure.figsize'] = (7, 5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400

__figdir__ = Path('../img/')
__figdir__.mkdir(parents=True, exist_ok=True)
savefig_args = {'bbox_inches': 'tight', 'pad_inches': 0.2}
plotfiletype = 'png'
savefig = False

# %% Load data
data_dir = home_dir / 'Python/SAFARI_mooring/data'

ds_flux = xr.open_dataset(data_dir / 'SAFARI_fluxes.nc')
ds_L3   = xr.open_dataset(data_dir / 'SAFARI_L3_met.nc')
ds_era5 = xr.open_dataset(data_dir / 'external/ERA5_surface_SAFARI_site_timeseries.nc').rename({'valid_time': 'time'})


# Zero or negative LW is a bad value (physically impossible); mask to NaN
ds_L3['longwave_radiation_downwards'] = ds_L3.longwave_radiation_downwards.where(
    ds_L3.longwave_radiation_downwards > 0)

# Clip ERA5 to SAFARI deployment period
t_start = ds_flux.time.values[0]
t_end   = ds_flux.time.values[-1]
ds_era5 = ds_era5.sel(time=slice(t_start, t_end))

# %% ERA5 radiation centering
# ERA5 accumulated radiation is labeled at the end of the 1-hour accumulation period
# (value at T = mean over (T-1h, T]). Shift back 30 min to center the timestamps.
# Kept separate from instantaneous variables so the shift does not affect them.
half_hour   = np.timedelta64(30, 'm')
rad_vars    = ['solar_radiation_downwards', 'longwave_radiation_downwards']
ds_era5_rad = ds_era5[rad_vars].assign_coords(time=ds_era5.time - half_hour)

# %% scatter_with_stats helper
def scatter_with_stats(ax, x, y, label, xlabel, ylabel, bias_units, n_bins=20):
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]
    if len(xv) < 2:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return
    sc = ax.scatter(xv, yv, label=label, alpha=0.4, s=10)
    lims = [min(xv.min(), yv.min()), max(xv.max(), yv.max())]
    ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.6)
    slope, intercept = np.polyfit(xv, yv, 1)
    r2   = np.corrcoef(xv, yv)[0, 1] ** 2
    bias = np.mean(yv - xv)
    rmsd = np.sqrt(np.mean((yv - xv) ** 2))
    ax.text(
        0.98, 0.02,
        f"bias={bias:.2f} {bias_units}\nRMSD={rmsd:.2f} {bias_units}\n"
        f"R$^2$={r2:.2f}\nfit: y={slope:.2f}x+{intercept:.2f}",
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
    )
    if n_bins and len(xv) >= n_bins:
        color = sc.get_facecolor()[0]
        bin_edges = np.linspace(xv.min(), xv.max(), n_bins + 1)
        bx, by, be = [], [], []
        for i in range(n_bins):
            in_bin = (xv >= bin_edges[i]) & (xv < bin_edges[i + 1])
            if i == n_bins - 1:
                in_bin = (xv >= bin_edges[i]) & (xv <= bin_edges[i + 1])
            vals = yv[in_bin]
            if len(vals) >= 2:
                bx.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
                by.append(np.mean(vals))
                be.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        if bx:
            ax.errorbar(bx, by, yerr=be, color='k', linewidth=1.5, capsize=3, zorder=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend(fontsize=8)

# %% Time series overview
fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

axs[0].plot(ds_flux.time, ds_flux.wind_speed_10_meters,              color='C0', label='SAFARI 10 m', linewidth=0.8)
axs[0].plot(ds_era5.time, ds_era5.wind_speed,                        color='C0', label='ERA5 10 m',   linewidth=0.8, linestyle='--')
axs[1].plot(ds_flux.time, ds_flux.air_temperature_at_reference_height, color='C1', label='SAFARI air 3 m',  linewidth=0.8)
axs[1].plot(ds_era5.time, ds_era5.air_temperature,                    color='C1', label='ERA5 air 2 m', linewidth=0.8, linestyle='--')
axs[1].plot(ds_L3.time,   ds_L3.sea_surface_temperature,              color='C4', label='SAFARI SST',    linewidth=0.8)
axs[1].plot(ds_era5.time, ds_era5.skin_temperature,                   color='C4', label='ERA5 skin T',   linewidth=0.8, linestyle='--')
axs[2].plot(ds_flux.time, ds_flux.relative_humidity_at_reference_height, color='C2', label='SAFARI 3 m', linewidth=0.8)
axs[2].plot(ds_era5.time, ds_era5.relative_humidity,                     color='C2', label='ERA5 2 m',   linewidth=0.8, linestyle='--')
axs[3].plot(ds_L3.time,       ds_L3.solar_radiation_downwards,           color='C3', label='SAFARI', linewidth=0.8)
axs[3].plot(ds_era5_rad.time, ds_era5_rad.solar_radiation_downwards,   color='C3', label='ERA5',              linewidth=0.8, linestyle='--')

axs[0].set_ylabel('Wind Speed (m/s)')
axs[1].set_ylabel('Air Temp. (°C)')
axs[2].set_ylabel('Rel. Hum. (%)')
axs[3].set_ylabel('Solar Rad. (W/m²)')
for a in axs:
    a.legend(fontsize=7)
    a.grid()
fig.autofmt_xdate()
plt.tight_layout()
if savefig:
    plt.savefig(__figdir__ / f'SAFARI-ERA5-timeseries.{plotfiletype}', **savefig_args)
    plt.close()

# %% Scatter: SAFARI vs ERA5
# Align ERA5 to SAFARI flux time grid
e5 = ds_era5.interp(time=ds_flux.time)

fig, axs = plt.subplots(3, 2, figsize=(7, 9))
fig.suptitle('ERA5 vs SAFARI (hourly)', fontsize=10)

scatter_with_stats(axs[0, 0], ds_flux.wind_speed_10_meters,               e5.wind_speed,
                   'Wind Speed',  'Buoy 10 m (m/s)',   'ERA5 10 m (m/s)',   'm/s')
scatter_with_stats(axs[0, 1], ds_flux.air_temperature_at_reference_height, e5.air_temperature,
                   'Air Temp',    'Buoy 3 m (°C)',     'ERA5 2 m (°C)',      '°C')
scatter_with_stats(axs[1, 0], ds_flux.relative_humidity_at_reference_height, e5.relative_humidity,
                   'Rel. Hum.',   'Buoy 3 m (%)',      'ERA5 2 m (%)',       '%')
scatter_with_stats(axs[1, 1], ds_flux.air_pressure_10_meters,              e5.barometric_pressure,
                   'Air Pressure','Buoy 10 m (mb)',    'ERA5 (mb)',          'mb')

# Radiation: align centered ERA5 to SAFARI L3 time grid
ds_era5_rad_aligned = ds_era5_rad.interp(time=ds_L3.time)
scatter_with_stats(axs[2, 0], ds_L3.solar_radiation_downwards,             ds_era5_rad_aligned.solar_radiation_downwards,
                   'Solar Rad.',  'Buoy (W/m²)',       'ERA5 (W/m²)',        'W/m²')
# SST is instantaneous — use unshifted ERA5
e5_L3 = ds_era5.interp(time=ds_L3.time)
scatter_with_stats(axs[2, 1], ds_L3.sea_surface_temperature,               e5_L3.skin_temperature,
                   'SST',         'Buoy SST(°C)',   'ERA5 skin temp (°C)', '°C')

plt.tight_layout()
if savefig:
    plt.savefig(__figdir__ / f'SAFARI-ERA5-scatter.{plotfiletype}', **savefig_args)
    plt.close()

# %% Scatter: longwave radiation and wave height
e5_wave = ds_era5.interp(time=ds_L3.time)
fig, axs = plt.subplots(1, 2, figsize=(7, 4))
scatter_with_stats(axs[0], ds_L3.longwave_radiation_downwards, ds_era5_rad_aligned.longwave_radiation_downwards,
                   'LW Rad.',    'Buoy (W/m²)', 'ERA5 (W/m²)', 'W/m²')
scatter_with_stats(axs[1], ds_L3.wave_height,                  e5_wave.wave_height,
                   'Wave Height','Buoy (m)',            'ERA5 (m)',    'm')
fig.suptitle('ERA5 vs SAFARI', fontsize=10)
plt.tight_layout()
if savefig:
    plt.savefig(__figdir__ / f'SAFARI-ERA5-lw-wave-scatter.{plotfiletype}', **savefig_args)
    plt.close()
