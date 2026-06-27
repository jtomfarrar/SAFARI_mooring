# 2026/03/26 Tom Farrar  plot Wirewalker up-profile data from 
# SAFARI mooring and save a netCDF file for other analysis

# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import os
import sys
import json
import ssl
import gsw
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# %%
# Set working directory
home_dir = Path.home()
os.chdir(home_dir / 'Python/SAFARI_mooring/src')

# %%
ip = get_ipython() if "get_ipython" in globals() else None
if ip is not None:
    ip.run_line_magic("matplotlib", "widget")

plt.rcParams['figure.figsize'] = (9, 5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400

__figdir__ = Path('../img/')
__figdir__.mkdir(parents=True, exist_ok=True)
savefig_args = {'bbox_inches': 'tight', 'pad_inches': 0.2}
plotfiletype = 'png'
savefig = False

# %%
# Nominal SAFARI mooring location
site_lat = 33.4
site_lon = -158

# %%
# Load Wirewalker profiles
ww_source_url = 'https://uop.whoi.edu/currentprojects/SAFARI/data/ww_up_profiles_qc.json'


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def load_json_url(url, timeout=30, verify_ssl=True):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        ctx = _ssl_context() if verify_ssl else ssl._create_unverified_context()
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.load(resp)
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"Failed to load JSON from {url}: {e}") from e


print('Loading Wirewalker profiles...')
ww = load_json_url(ww_source_url, verify_ssl=False)
print(f'Loaded {len(ww)} profiles')

# %%
# Parse into 2-D arrays (depth x time)
depth = np.linspace(0, 300, 153)   # 153 depth bins, 0-300 m

n_profiles = len(ww)
n_depth = len(depth)

time = np.array([dt.datetime.strptime(p['time'], '%Y-%m-%d %H:%M:%S') for p in ww])

T_mat = np.full((n_depth, n_profiles), np.nan)
S_mat = np.full((n_depth, n_profiles), np.nan)

for j, p in enumerate(ww):
    T_mat[:, j] = [v if v is not None else np.nan for v in p['temperature']]
    S_mat[:, j] = [v if v is not None else np.nan for v in p['salinity']]

print(f'Time range: {time[0]} to {time[-1]}')
print(f'T range: {np.nanmin(T_mat):.2f} to {np.nanmax(T_mat):.2f} °C')
print(f'S range: {np.nanmin(S_mat):.2f} to {np.nanmax(S_mat):.2f}')

# %%
# Estimate buoyancy frequency
pressure = gsw.p_from_z(-depth, site_lat)
pressure_mat = np.tile(pressure[:, np.newaxis], (1, n_profiles))
lon_mat = np.full_like(T_mat, site_lon, dtype=float)
lat_mat = np.full_like(T_mat, site_lat, dtype=float)

SA_mat = gsw.SA_from_SP(S_mat, pressure_mat, lon_mat, lat_mat)
CT_mat = gsw.CT_from_t(SA_mat, T_mat, pressure_mat)
N2_mat, pressure_mid = gsw.Nsquared(SA_mat, CT_mat, pressure_mat, lat=site_lat, axis=0)
N2_plot_mat = (
    xr.DataArray(N2_mat, dims=('depth_mid', 'time'))
    .rolling(time=3, center=True, min_periods=1)
    .mean()
    .values
)
depth_mid = 0.5 * (depth[:-1] + depth[1:])
pressure_mid = pressure_mid[:, 0]

print(f'N2 range: {np.nanmin(N2_mat):.2e} to {np.nanmax(N2_mat):.2e} s^-2')

# %%
# Temperature depth-time plot
fig, ax = plt.subplots(figsize=(9, 4))
pcm = ax.pcolormesh(time, depth, T_mat, cmap='RdYlBu_r', shading='auto')
ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_title('SAFARI Wirewalker — Temperature (°C)')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
fig.autofmt_xdate()
plt.colorbar(pcm, ax=ax, label='°C')
plt.tight_layout()
if savefig:
    plt.savefig(__figdir__ / f'SAFARI_WW_temperature.{plotfiletype}', **savefig_args)


# %%
# Salinity depth-time plot
fig, ax = plt.subplots(figsize=(9, 4))
pcm = ax.pcolormesh(time, depth, S_mat, cmap='viridis', shading='auto')
ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_title('SAFARI Wirewalker — Salinity')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
fig.autofmt_xdate()
plt.colorbar(pcm, ax=ax, label='PSU')
plt.tight_layout()
if savefig:
    plt.savefig(__figdir__ / f'SAFARI_WW_salinity.{plotfiletype}', **savefig_args)

# %%
# Buoyancy frequency squared depth-time plot
from matplotlib.colors import LogNorm

fig, ax = plt.subplots(figsize=(9, 4))
N2_plot_positive = np.where(N2_plot_mat > 0, N2_plot_mat, np.nan)
n2_vmin, n2_vmax = np.nanpercentile(N2_plot_positive, [0.1, 99.9])
norm = LogNorm(vmin=n2_vmin, vmax=n2_vmax)
pcm = ax.pcolormesh(time, depth_mid, N2_plot_positive, cmap='viridis', norm=norm, shading='auto')
ax.invert_yaxis()
ax.set_ylabel('Depth (m)')
ax.set_title('SAFARI Wirewalker — Buoyancy Frequency Squared, 3-Profile Running Mean')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
fig.autofmt_xdate()
plt.colorbar(pcm, ax=ax, label='N$^2$ (s$^{-2}$)')
plt.tight_layout()
if savefig:
    plt.savefig(__figdir__ / f'SAFARI_WW_N2.{plotfiletype}', **savefig_args)


# %%
# Combined T and S depth-time plot
fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

pcm0 = axs[0].pcolormesh(time, depth, T_mat, cmap='RdYlBu_r', shading='auto')
axs[0].invert_yaxis()
axs[0].set_ylabel('Depth (m)')
axs[0].set_title('Temperature (°C)')
plt.colorbar(pcm0, ax=axs[0], label='°C')

pcm1 = axs[1].pcolormesh(time, depth, S_mat, cmap='viridis', shading='auto')
axs[1].invert_yaxis()
axs[1].set_ylabel('Depth (m)')
axs[1].set_title('Salinity')
plt.colorbar(pcm1, ax=axs[1], label='PSU')

axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
axs[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
fig.autofmt_xdate()
fig.suptitle('SAFARI Wirewalker Profiles')
plt.tight_layout()
if savefig:
    plt.savefig(__figdir__ / f'SAFARI_WW_T_S.{plotfiletype}', **savefig_args)


# %%
# Combined T and S depth-time plot, upper 100 m
fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

pcm0 = axs[0].pcolormesh(time, depth, T_mat, cmap='RdYlBu_r', vmin=14.5, vmax=21, shading='auto')
axs[0].set_ylim(100, 0)
axs[0].set_ylabel('Depth (m)')
axs[0].set_title('Temperature (°C)')
plt.colorbar(pcm0, ax=axs[0], label='°C')

pcm1 = axs[1].pcolormesh(time, depth, S_mat, cmap='viridis', vmin=34.3, vmax=34.65, shading='auto')
axs[1].set_ylim(100, 0)
axs[1].set_ylabel('Depth (m)')
axs[1].set_title('Salinity')
plt.colorbar(pcm1, ax=axs[1], label='PSU')

axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
axs[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
fig.autofmt_xdate()
fig.suptitle('SAFARI Wirewalker Profiles, Upper 100 m')
plt.tight_layout()
if savefig:
    plt.savefig(__figdir__ / f'SAFARI_WW_T_S_upper100.{plotfiletype}', **savefig_args)


# %%
# Write Wirewalker profiles to NetCDF
data_dir = Path('../data/')
data_dir.mkdir(parents=True, exist_ok=True)

ds_ww = xr.Dataset(
    data_vars={
        'temperature': (('depth', 'time'), T_mat,
                        {'long_name': 'sea water temperature', 'units': 'degree_C'}),
        'salinity': (('depth', 'time'), S_mat,
                     {'long_name': 'practical salinity', 'units': '1'}),
        'absolute_salinity': (('depth', 'time'), SA_mat,
                              {'long_name': 'absolute salinity', 'units': 'g kg-1'}),
        'conservative_temperature': (('depth', 'time'), CT_mat,
                                     {'long_name': 'conservative temperature', 'units': 'degree_C'}),
        'buoyancy_frequency_squared': (('depth_mid', 'time'), N2_mat,
                                       {'long_name': 'buoyancy frequency squared', 'units': 's-2'}),
    },
    coords={
        'time': ('time', time, {'long_name': 'profile time'}),
        'depth': ('depth', depth, {'long_name': 'depth', 'units': 'm', 'positive': 'down'}),
        'depth_mid': ('depth_mid', depth_mid, {'long_name': 'midpoint depth', 'units': 'm', 'positive': 'down'}),
        'pressure': ('depth', pressure, {'long_name': 'sea pressure', 'units': 'dbar'}),
        'pressure_mid': ('depth_mid', pressure_mid, {'long_name': 'midpoint sea pressure', 'units': 'dbar'}),
    },
    attrs={
        'title': 'SAFARI Wirewalker up-profile data',
        'source': ww_source_url,
        'latitude': site_lat,
        'longitude': site_lon,
        'history': (
            f'Created {dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")} '
            'by plot_SAFARI_wirewalker.py'
        ),
    },
)

encoding = {}
for var in list(ds_ww.data_vars) + list(ds_ww.coords):
    enc = {'zlib': True, 'complevel': 4}
    if np.issubdtype(ds_ww[var].dtype, np.floating):
        enc['dtype'] = 'float32'
        enc['_FillValue'] = np.nan
    encoding[var] = enc

ww_netcdf_file = data_dir / 'SAFARI_wirewalker_profiles.nc'
ds_ww.to_netcdf(ww_netcdf_file, encoding=encoding)
print(f'Wrote {ww_netcdf_file}')
