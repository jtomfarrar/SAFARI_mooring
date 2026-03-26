
# 2026/03/26 Tom Farrar  plot Wirewalker up-profile data from SAFARI mooring

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import os
import sys
import json
import ssl
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

# %%
# Set working directory
home_dir = Path.home()
os.chdir(home_dir / 'Python/SAFARI_mooring/src')

# %%
%matplotlib ipympl
plt.rcParams['figure.figsize'] = (9, 5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400

__figdir__ = Path('../img/')
__figdir__.mkdir(parents=True, exist_ok=True)
savefig_args = {'bbox_inches': 'tight', 'pad_inches': 0.2}
plotfiletype = 'png'
savefig = False

# %%
# Load Wirewalker profiles
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
ww = load_json_url('https://uop.whoi.edu/currentprojects/SAFARI/data/ww_up_profiles_qc.json', verify_ssl=False)
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
# Hovmoller plot: Temperature
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
plt.close()

# %%
# Hovmoller plot: Salinity
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
# Combined T and S Hovmoller
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

