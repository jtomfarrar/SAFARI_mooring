# %% SAFARI_mooring - ERA5 daily climatology at the SAFARI mooring site
# Created: 2026-06-23

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
ip = get_ipython() if "get_ipython" in globals() else None
if ip is not None:
    ip.run_line_magic("matplotlib", "widget")

plt.rcParams['figure.figsize'] = (7, 5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400

__figdir__ = Path('../img/')
__figdir__.mkdir(parents=True, exist_ok=True)
savefig_args = {'bbox_inches': 'tight', 'pad_inches': 0.2}
plotfiletype = 'png'
savefig = True

# %% Plot configuration
site_name = 'SAFARI'
data_dir = home_dir / 'Python/SAFARI_mooring/data'
era5_file = data_dir / 'external/ERA5_surface_SAFARI_site_timeseries.nc'

plot_specs = [
    ('sea_surface_temperature', 'ERA5 SST', 'SST (degC)'),
    ('wave_height', 'ERA5 Significant Wave Height', 'Sig. Wave Ht. (m)'),
    ('wind_speed', 'ERA5 Wind Speed', 'Wind Speed (m/s)'),
    ('air_temperature', 'ERA5 Air Temperature', 'Air Temp. (degC)'),
    ('relative_humidity', 'ERA5 Relative Humidity', 'Rel. Hum. (%)'),
]

annual_plot_specs = [
    ('sea_surface_temperature', 'SST', 'SST (degC)'),
    ('air_temperature', 'Air Temperature', 'Air Temp. (degC)'),
    ('specific_humidity', 'Specific Humidity', 'Spec. Hum. (g/kg)'),
    ('wind_speed', 'Wind Speed', 'Wind Speed (m/s)'),
    ('wave_height', 'Significant Wave Height', 'Sig. Wave Ht. (m)'),
]

climatology_style = {'color': 'k', 'linewidth': 1.7, 'linestyle': '--', 'alpha': 0.9}
daily_style = {'color': '0.45', 'linewidth': 0.7, 'alpha': 0.65}
highlight_years = {
    2024: {'color': 'C3', 'linewidth': 2.2, 'zorder': 4},
    2025: {'color': 'C0', 'linewidth': 2.2, 'zorder': 5},
    2026: {'color': 'C1', 'linewidth': 2.4, 'zorder': 6},
}
month_ticks = pd.date_range('2001-01-01', '2001-12-01', freq='MS')

# %% Helper functions
def drop_feb29(ds):
    is_feb29 = (ds.time.dt.month == 2) & (ds.time.dt.day == 29)
    return ds.where(~is_feb29, drop=True)


def add_month_day_coord(ds):
    month_day = ds.time.dt.strftime('%m-%d')
    return ds.assign_coords(month_day=('time', month_day.data))


def add_specific_humidity(ds):
    air_temperature = ds['air_temperature']
    relative_humidity = ds['relative_humidity']
    pressure = ds['barometric_pressure']
    saturation_vapor_pressure = 6.112 * np.exp((17.625 * air_temperature) / (243.04 + air_temperature))
    vapor_pressure = relative_humidity / 100 * saturation_vapor_pressure
    mixing_ratio = 0.622 * vapor_pressure / (pressure - vapor_pressure)
    specific_humidity = 1000 * mixing_ratio / (1 + mixing_ratio)
    specific_humidity.attrs['units'] = 'g kg-1'
    specific_humidity.attrs['long_name'] = 'specific humidity'
    return ds.assign(specific_humidity=specific_humidity)


def compute_daily_climatology(ds, var_names):
    ds_daily = ds[var_names].resample(time='1D').mean(skipna=True)
    ds_daily = drop_feb29(ds_daily)
    ds_daily = add_month_day_coord(ds_daily)
    climatology_ds = ds_daily.groupby('month_day').mean(skipna=True)

    climatology_time = np.array([np.datetime64(f'2001-{month_day}') for month_day in climatology_ds.month_day.values])
    climatology_ds = climatology_ds.assign_coords(climatology_time=('month_day', climatology_time))
    climatology_ds.attrs['site_name'] = site_name
    climatology_ds.attrs['site_longitude'] = float(ds.longitude.values)
    climatology_ds.attrs['site_latitude'] = float(ds.latitude.values)
    return ds_daily, climatology_ds


def repeat_daily_climatology(ds_daily, climatology_ds):
    ds_daily = add_month_day_coord(drop_feb29(ds_daily))
    repeated_ds = xr.Dataset(coords={'time': ds_daily.time})

    for var_name in climatology_ds.data_vars:
        climatology_series = climatology_ds[var_name].to_series()
        repeated_values = ds_daily.month_day.to_series().map(climatology_series).values
        repeated_ds[var_name] = xr.DataArray(
            repeated_values,
            coords={'time': ds_daily.time},
            dims=('time',),
            attrs=climatology_ds[var_name].attrs,
        )

    repeated_ds.attrs['site_name'] = site_name
    repeated_ds.attrs['site_longitude'] = climatology_ds.attrs['site_longitude']
    repeated_ds.attrs['site_latitude'] = climatology_ds.attrs['site_latitude']
    return repeated_ds


def format_climatology_axis(ax):
    ax.set_xlim(np.datetime64('2001-01-01'), np.datetime64('2001-12-31'))
    ax.set_xticks(month_ticks)
    ax.set_xticklabels([t.strftime('%b') for t in month_ticks])
    ax.grid(True, color='0.85')


# %% Load ERA5 time series
ds_era5 = xr.open_dataset(era5_file).rename({'valid_time': 'time'})
ds_era5 = add_specific_humidity(ds_era5)
var_names = [var_name for var_name, title, ylabel in plot_specs]
annual_var_names = [var_name for var_name, title, ylabel in annual_plot_specs]
var_names = list(dict.fromkeys(var_names + annual_var_names))
ds_daily, climatology_ds = compute_daily_climatology(ds_era5, var_names)
repeated_climatology_ds = repeat_daily_climatology(ds_daily, climatology_ds)

print(f'Loaded ERA5 time series: {era5_file}')
print(f'Daily ERA5 record: {str(ds_daily.time.values[0])[:10]} to {str(ds_daily.time.values[-1])[:10]}')
print(f'Daily climatology days: {len(climatology_ds.month_day)}')

# %% Plot time series with repeated annual climatology
fig, axs = plt.subplots(len(plot_specs), 1, figsize=(7, 10.5), sharex=True)
fig.suptitle(f'ERA5 daily conditions at {site_name}', fontsize=10)

for ax, (var_name, title, ylabel) in zip(axs, plot_specs):
    ax.plot(ds_daily.time, ds_daily[var_name], label='daily mean', **daily_style)
    ax.plot(repeated_climatology_ds.time, repeated_climatology_ds[var_name], label='daily climatology', **climatology_style)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, color='0.85')

fig.autofmt_xdate()
plt.tight_layout()
if savefig:
    plt.savefig(__figdir__ / f'{site_name}-ERA5-daily-timeseries-climatology.{plotfiletype}', **savefig_args)

# %% Plot annual daily climatology
fig, axs = plt.subplots(len(plot_specs), 1, figsize=(7, 10.5), sharex=True)
fig.suptitle(f'ERA5 annual daily climatology at {site_name}', fontsize=10)

for ax, (var_name, title, ylabel) in zip(axs, plot_specs):
    ax.plot(climatology_ds.climatology_time, climatology_ds[var_name], color='C0', linewidth=1.5)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel)
    format_climatology_axis(ax)

plt.tight_layout()
if savefig:
    plt.savefig(__figdir__ / f'{site_name}-ERA5-daily-climatology.{plotfiletype}', **savefig_args)

# %% Plot annual evolution by year
fig, axs = plt.subplots(len(annual_plot_specs), 1, figsize=(7, 10.5), sharex=True)
fig.suptitle(f'ERA5 annual evolution at {site_name}', fontsize=10)
years = np.unique(ds_daily.time.dt.year.values)

for ax, (var_name, title, ylabel) in zip(axs, annual_plot_specs):
    daily_var = ds_daily[var_name]

    for year in years:
        var_year = daily_var.sel(time=daily_var.time.dt.year == year)
        var_year = drop_feb29(var_year.to_dataset(name=var_name))[var_name]
        var_year_time = pd.to_datetime(var_year.time.values)
        plot_time = pd.to_datetime([f'2001-{time_val.month:02d}-{time_val.day:02d}' for time_val in var_year_time])

        if int(year) in highlight_years:
            ax.plot(plot_time, var_year, label=str(int(year)), **highlight_years[int(year)])
        else:
            label = 'other years' if year == years[0] else None
            ax.plot(plot_time, var_year, color='0.75', linewidth=0.7, alpha=0.45, label=label, zorder=1)

    ax.plot(climatology_ds.climatology_time, climatology_ds[var_name], color='k', linewidth=1.7, linestyle='--',
            label='climatology', zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel)
    format_climatology_axis(ax)

axs[0].legend(fontsize=7, loc='best', ncol=2)
plt.tight_layout()
if savefig:
    plt.savefig(__figdir__ / f'{site_name}-ERA5-annual-evolution-by-year.{plotfiletype}', **savefig_args)
