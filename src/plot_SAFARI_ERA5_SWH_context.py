# %% SAFARI_mooring - Plot ERA5 maximum significant wave height over the SAFARI map domain
# Created: 2026-06-28

# %%
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import sys
from pathlib import Path
from mpl_toolkits.basemap import Basemap

# %%
# Set working directory
home_dir = Path.home()
os.chdir(home_dir / 'Python/SAFARI_mooring/src')

# %%
ip = get_ipython() if "get_ipython" in globals() else None
if ip is not None:
    ip.run_line_magic("matplotlib", "widget")

plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 400

__figdir__ = Path('../img/')
__figdir__.mkdir(parents=True, exist_ok=True)
savefig_args = {'bbox_inches': 'tight', 'pad_inches': 0.2}
plotfiletype = 'png'
savefig = True

# %% Plot configuration
site_name = 'SAFARI_2025_2026'
data_dir = home_dir / 'Python/SAFARI_mooring/data'
waves_file = data_dir / 'external/ERA5_surface_SAFARI_2020_2025_waves_202001_202512.nc'
l3_file = data_dir / 'SAFARI_L3_met.nc'

# Match the SAFARI_2025_2026 map domain in ERA5_extraction/src/ERA5_SAFARI_plots.py
lon0 = -161
lat0 = 35
lon_pt = -158
lat_pt = 33.44
dx = 48
dy = 25
map_resolution = 'l'

lon_min = lon0 - dx
lon_max = lon0 + dx
lat_min = lat0 - dy
lat_max = lat0 + dy

swh_levels = np.arange(0, 17, 0.5)

# %% Helper functions
def mean_mooring_location(l3_file, fallback_lon, fallback_lat):
    if not l3_file.exists():
        print(f'L3 file not found; using fallback mooring location: {fallback_lat:.4f}N, {fallback_lon:.4f}E')
        return fallback_lon, fallback_lat

    ds = xr.open_dataset(l3_file)
    mooring_lon = float(ds.longitude.mean(skipna=True))
    mooring_lat = float(ds.latitude.mean(skipna=True))
    ds.close()

    if not np.isfinite(mooring_lon) or not np.isfinite(mooring_lat):
        print(f'L3 position data are not finite; using fallback mooring location: {fallback_lat:.4f}N, {fallback_lon:.4f}E')
        return fallback_lon, fallback_lat

    print(f'Using SAFARI mean mooring location: {mooring_lat:.4f}N, {mooring_lon:.4f}E')
    return mooring_lon, mooring_lat


def add_map_grid(map_obj):
    map_obj.drawparallels(range(-90, 91, 15), labels=[1, 0, 0, 0], linewidth=0.4, color='0.45', dashes=[2, 2])
    map_obj.drawmeridians(range(-360, 361, 15), labels=[0, 0, 0, 1], linewidth=0.4, color='0.45', dashes=[2, 2])


def setup_swh_map():
    fig = plt.figure(figsize=(8, 5))
    map_obj = Basemap(
        projection='cyl',
        lat_1=lat0 - 5,
        lat_2=lat0 + 5,
        lat_0=lat0,
        lon_0=lon0,
        llcrnrlat=lat_min,
        urcrnrlat=lat_max,
        llcrnrlon=lon_min,
        urcrnrlon=lon_max,
        resolution=map_resolution,
    )
    return fig, map_obj


# %% Load data
lon_pt, lat_pt = mean_mooring_location(l3_file, lon_pt, lat_pt)
ds_waves = xr.open_dataset(waves_file, engine='netcdf4')
swh_max = ds_waves.swh.max(dim='valid_time', skipna=True)
swh_site_max = float(swh_max.interp(longitude=lon_pt, latitude=lat_pt))

print(f'ERA5 wave domain: {float(ds_waves.longitude.min()):.2f} to {float(ds_waves.longitude.max()):.2f}E, '
      f'{float(ds_waves.latitude.min()):.2f} to {float(ds_waves.latitude.max()):.2f}N')
print(f'Max SWH range: {float(swh_max.min()):.2f} to {float(swh_max.max()):.2f} m')
print(f'SAFARI mooring max SWH from ERA5: {swh_site_max:.1f} m')

# %% Plot maximum SWH map
lonmesh, latmesh = np.meshgrid(ds_waves.longitude.values, ds_waves.latitude.values)

fig, map = setup_swh_map()
x, y = map(lonmesh, latmesh)
cs = map.contourf(x, y, swh_max.values, levels=swh_levels, cmap='coolwarm', extend='max')
map.fillcontinents(color='0.75', lake_color='0.85', zorder=3)
map.drawcoastlines(linewidth=0.5, color='0.15', zorder=4)
map.drawcountries(linewidth=0.4, color='0.25', zorder=4)
add_map_grid(map)
map.colorbar(cs, location='right', size='5%', pad='2%', label='Max SWH (m)')

xpt, ypt = map(lon_pt, lat_pt)
map.plot(xpt, ypt, marker='D', color='m', markeredgecolor='k', markersize=7, linestyle='none', zorder=5)
plt.text(xpt + 1.2, ypt + 1.0, f'SAFARI mooring, \n {swh_site_max:.1f} m', color='m', fontsize=9, weight='bold', zorder=5)

plt.title('ERA5 Maximum Significant Wave Height (2020-2025)')
plt.tight_layout()

if savefig:
    plt.savefig(__figdir__ / f'SAFARI_ERA5_max_SWH_map.{plotfiletype}', **savefig_args)

# %%
ds_waves.close()
