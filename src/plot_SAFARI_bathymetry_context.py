# %% SAFARI_mooring - Plot GEBCO bathymetry over the SAFARI ERA5 map domain
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
import cmocean

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
gebco_file = Path('/mnt/d/tom_data/GEBCO/GEBCO_2026.nc')
l3_file = data_dir / 'SAFARI_L3_met.nc'

# Match the SAFARI_2025_2026 map domain in ERA5_extraction/src/ERA5_SAFARI_plots.py
lon0 = -161
lat0 = 35
lon_pt = -158
lat_pt = 33.44
dx = 48
dy = 25
map_resolution = 'l'
gebco_coarsen_factor = 12

lon_min = lon0 - dx
lon_max = lon0 + dx
lat_min = lat0 - dy
lat_max = lat0 + dy

depth_levels = np.arange(2000, 6600, 300)
bathymetry_cmap = cmocean.cm.deep

dutch_harbor_lon = -166.54
dutch_harbor_lat = 53.89
honolulu_lon = -157.86
honolulu_lat = 21.31

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


def subset_gebco_dateline(ds_gebco, lon_min, lon_max, lat_min, lat_max):
    lat_slice = slice(lat_min, lat_max)

    if lon_min < -180 and lon_max <= 180:
        west = ds_gebco.elevation.sel(lon=slice(lon_min + 360, 180), lat=lat_slice)
        west = west.assign_coords(lon=west.lon - 360)
        east = ds_gebco.elevation.sel(lon=slice(-180, lon_max), lat=lat_slice)
        elevation = xr.concat([west, east], dim='lon').sortby('lon')
    else:
        elevation = ds_gebco.elevation.sel(lon=slice(lon_min, lon_max), lat=lat_slice)

    if gebco_coarsen_factor > 1:
        elevation = elevation.coarsen(lat=gebco_coarsen_factor, lon=gebco_coarsen_factor, boundary='trim').mean()

    return elevation


def add_map_grid(map_obj):
    map_obj.drawparallels(range(-90, 91, 15), labels=[1, 0, 0, 0], linewidth=0.4, color='0.45', dashes=[2, 2])
    map_obj.drawmeridians(range(-360, 361, 15), labels=[0, 0, 0, 1], linewidth=0.4, color='0.45', dashes=[2, 2])


def setup_bathymetry_map(depth):
    lonmesh, latmesh = np.meshgrid(depth.lon.values, depth.lat.values)

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

    x, y = map_obj(lonmesh, latmesh)
    cs = map_obj.contourf(x, y, depth.values, levels=depth_levels, cmap=bathymetry_cmap, extend='both')
    map_obj.fillcontinents(color='0.75', lake_color='0.85', zorder=3)
    map_obj.drawcoastlines(linewidth=0.5, color='0.15', zorder=4)
    map_obj.drawcountries(linewidth=0.4, color='0.25', zorder=4)
    add_map_grid(map_obj)
    map_obj.colorbar(cs, location='right', size='5%', pad='2%', label='Depth (m)')
    return fig, map_obj


# %% Load data
lon_pt, lat_pt = mean_mooring_location(l3_file, lon_pt, lat_pt)
ds_gebco = xr.open_dataset(gebco_file, engine='netcdf4')
elevation = subset_gebco_dateline(ds_gebco, lon_min, lon_max, lat_min, lat_max)
depth = (-elevation).where(elevation < 0)
mooring_depth = float(depth.interp(lon=lon_pt, lat=lat_pt))

print(f'GEBCO subset: {float(depth.lon.min()):.2f} to {float(depth.lon.max()):.2f}E, '
      f'{float(depth.lat.min()):.2f} to {float(depth.lat.max()):.2f}N')
print(f'GEBCO plot grid: {depth.sizes["lat"]} lat x {depth.sizes["lon"]} lon')
print(f'SAFARI mooring depth from GEBCO: {mooring_depth:.0f} m')

# %% Plot bathymetry map
fig, map = setup_bathymetry_map(depth)
xpt, ypt = map(lon_pt, lat_pt)
map.plot(xpt, ypt, marker='D', color='m', markeredgecolor='k', markersize=7, linestyle='none', zorder=5)
plt.text(xpt + 1.2, ypt + 1.0, f'SAFARI mooring, \n {mooring_depth:.0f} m', color='m', fontsize=12, weight='bold', zorder=5)

plt.title(f'GEBCO 2026 Bathymetry')
plt.tight_layout()

if savefig:
    plt.savefig(__figdir__ / f'SAFARI_GEBCO_2026_bathymetry_map.{plotfiletype}', **savefig_args)

# %% Plot bathymetry map with Dutch Harbor-Hawaii line
fig, map = setup_bathymetry_map(depth)
x_route, y_route = map([dutch_harbor_lon, honolulu_lon], [dutch_harbor_lat, honolulu_lat])
map.plot(x_route, y_route, color='m', linestyle='--', linewidth=2, zorder=5)
map.plot(x_route, y_route, marker='o', color='m', markeredgecolor='k', markersize=5, linestyle='none', zorder=6)
plt.text(x_route[0] + 1.0, y_route[0] - 1.5, 'Dutch Harbor', color='m', fontsize=12, weight='bold', va='top', zorder=6)
plt.text(x_route[1] + 3.0, y_route[1] - 1.4, 'Honolulu', color='m', fontsize=12, weight='bold', zorder=6)

plt.title(f'GEBCO 2026 Bathymetry')
plt.tight_layout()

if savefig:
    plt.savefig(__figdir__ / f'SAFARI_GEBCO_2026_bathymetry_Dutch_Harbor_Hawaii.{plotfiletype}', **savefig_args)

# %% Plot bathymetry map with Dutch Harbor-Hawaii line
fig, map = setup_bathymetry_map(depth)
x_route, y_route = map([dutch_harbor_lon, honolulu_lon], [dutch_harbor_lat, honolulu_lat])
map.plot(x_route, y_route, color='m', linestyle='--', linewidth=2, zorder=5)
map.plot(x_route, y_route, marker='o', color='m', markeredgecolor='k', markersize=5, linestyle='none', zorder=6)
plt.text(x_route[0] + 1.0, y_route[0] - 1.5, 'Dutch Harbor', color='m', fontsize=12, weight='bold', va='top', zorder=6)
plt.text(x_route[1] + 3.0, y_route[1] - 1.4, 'Honolulu', color='m', fontsize=12, weight='bold', zorder=6)
x_safari, y_safari = map(lon_pt, lat_pt)
map.plot(x_safari, y_safari, marker='D', color='m', markeredgecolor='k', markersize=7, linestyle='none', zorder=5)
plt.text(x_safari + 1.2, y_safari + 1.0, f'SAFARI mooring, \n {mooring_depth:.0f} m', color='m', fontsize=12, weight='bold', zorder=5)

plt.title(f'GEBCO 2026 Bathymetry')
plt.tight_layout()

if savefig:
    plt.savefig(__figdir__ / f'SAFARI_GEBCO_2026_bathymetry_Dutch_Harbor_Hawaii_SAFARI_site.{plotfiletype}', **savefig_args)

# %%
ds_gebco.close()
