# %% [markdown]
# # Download, plot near-real-time DUACS SSH product
# 
# Tom Farrar, started 10/9/2022
# 
# * Download with motuclient
# * Plot latest map
# * make movie of longer time
# * extract U, V time series at some point

# %%
# mamba install conda-forge::copernicusmarine --yes
import copernicusmarine
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import cartopy.crs as ccrs                   # import projections
import cartopy
import cartopy.mpl.ticker as cticker
import gsw
import pandas as pd

# import cftime


# %%
# Change to this directory
home_dir = os.path.expanduser("~")

# To work for Tom and other people; other users can add elif statements for their own directory structure
if os.path.exists(home_dir + '/Python/SAFARI_mooring/src'):
    os.chdir(home_dir + '/Python/SAFARI_mooring/src')

# %%
%matplotlib widget
plt.rcParams['figure.figsize'] = (5,4)
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 400
plt.close('all')

__figdir__ = '../img/' + 'SSH_plots/'
savefig_args = {'bbox_inches':'tight', 'pad_inches':0.2}
plotfiletype='png'

# %%
# Nominal mooring location: 
mooring = dict(
    lon = [-158],
    lat = [33.4])
mooring['lon'][0] = (mooring['lon'][0] + 360) % 360
# %%
savefig = False
zoom = False
domovie = False
if zoom:
    dx = 7.5
    dy = 7.5
    xmin, xmax = (mooring['lon'][0] - dx, mooring['lon'][0] + dx)
    ymin, ymax = (mooring['lat'][0] - dy, mooring['lat'][0] + dy)
    levels = np.linspace(-.2,.2,11)
else:
    dx = 30
    dy = 30
    xmin, xmax = (mooring['lon'][0] - dx, mooring['lon'][0] + dx)
    ymin, ymax = (mooring['lat'][0] - dy, mooring['lat'][0] + dy)
    levels = np.linspace(-.3,.3,21)

# %%
# Run the following line once to create the login file for Copernicus Marine Service
# copernicusmarine.login()

# %%
if not os.path.exists('../data/external/aviso.nc'):
    print('Need to download the data')
    ds = copernicusmarine.open_dataset(
        dataset_id = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
        minimum_longitude = xmin, maximum_longitude = xmax,
        minimum_latitude = ymin, maximum_latitude = ymax,
        minimum_depth = 0., maximum_depth = 10., 
        start_datetime = "2025-11-15 00:00:00",    
        end_datetime = "2026-05-15 23:59:59", 
        variables = ['adt', 'sla', 'ugos', 'vgos'], 
        )
    ds.to_netcdf('../data/external/aviso.nc')
else:
    ds = xr.open_dataset('../data/external/aviso.nc')
    # check when that file was written
    print('File exists, check the date')
    print('File created: ', ds.time[0].values)
    print('File last modified: ', ds.time[-1].values)
    
    # download a small set of the data to see if there are any problems
    last_time = str(pd.to_datetime(ds.time[-1].values))
    ds_test = copernicusmarine.open_dataset(
        dataset_id = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D",
        minimum_longitude = xmin, maximum_longitude = xmax,
        minimum_latitude = ymin, maximum_latitude = ymax,
        minimum_depth = 0., maximum_depth = 10., 
        start_datetime = last_time,    
        end_datetime = "2026-05-16 23:59:59", 
        variables = ['adt', 'sla', 'ugos', 'vgos'], 
        )
    # check if the last time of the new file is more recent than the last time of the old file
    if ds_test.time[-1].values > ds.time[-1].values:
        print('New data is more recent, appending to the old file')
        # append the new data to the old dataset
        ds = xr.concat([ds, ds_test], dim='time')
        # Check if the ugos at the last time is all nans
        if np.all(np.isnan(ds.ugos.isel(time=-1))):
            print('Last time is all nans, removing it')
            ds = ds.isel(time=slice(0, -1))
        ds.to_netcdf('../data/external/aviso.nc')
    else:
        print('Existing data is up to date, no need to append.')
        # ds_test.to_netcdf('../data/external/aviso.nc', mode='a')

# %%


# %%
def plot_SSH_map(tind):
    plt.clf()
    ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=mooring['lon'][0]))  # Orthographic
    extent = [xmin, xmax, ymin, ymax]
    day_str = np.datetime_as_string(ds.time[tind], unit='D')
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_title('Sea level anomaly (DUACS), '+ day_str,size = 10.)

    #plt.set_cmap(cmap=plt.get_cmap('nipy_spectral'))
    plt.set_cmap(cmap=plt.get_cmap('turbo'))
    # Use explicit tick locations and matching gridlines
    xlocs = np.arange(170, 233, 15)
    ylocs = np.arange(10, 70, 10)
    ax.set_xticks(xlocs, crs=ccrs.PlateCarree())
    ax.set_yticks(ylocs, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter(number_format='.0f'))
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter(number_format='.0f'))
    #gl.xlocator = matplotlib.ticker.FixedLocator(np.arange(170, 233, 10))

    #gl.xlocator = matplotlib.ticker.MaxNLocator(10)
    #gl.xlocator = matplotlib.ticker.AutoLocator()
    #gl.xlocator = matplotlib.ticker.FixedLocator(np.arange(0, 360 ,15))

    cs = ax.contourf(ds.longitude,ds.latitude,np.squeeze(ds.sla.isel(time=tind)), levels, extend='both', transform=ccrs.PlateCarree())
    cb = plt.colorbar(cs,fraction = 0.022,extend='both')
    cb.set_label('SLA [m]',fontsize = 10)
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, zorder=3, facecolor=[.6,.6,.6], edgecolor='black')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, x_inline=False, y_inline=False,
                      alpha=0.7, linestyle='--', xlocs=xlocs, ylocs=ylocs, zorder=4)
   
    # Add a 10 km scale bar
    km_per_deg_lat=gsw.geostrophy.distance((mooring['lon'][0],mooring['lon'][0]), (mooring['lat'][0],mooring['lat'][0]+1))/1000
    deg_lat_equal_10km=10/km_per_deg_lat
    x0 = mooring['lon'][0] - dx + 2 
    y0 = mooring['lat'][0] - dy + 2
    ax.plot(x0+np.asarray([0, 0]),y0+np.asarray([0.,deg_lat_equal_10km[0]]),transform=ccrs.PlateCarree(),color='k',zorder=3)
    ax.text(x0+1/60, y0+.15/60, '10 km', fontsize=6,transform=ccrs.PlateCarree())

    u = np.squeeze(ds.ugos.isel(time=tind)) #dtype=object
    v = np.squeeze(ds.vgos.isel(time=tind))
    skip = 5
    scalefac = 10
    x0 = mooring['lon'][0] - dx + 2
    y0 = mooring['lat'][0] - dy + 1.5
    ax.plot(mooring['lon'], mooring['lat'], marker='o', color='k', transform=ccrs.PlateCarree(), zorder=3)
    ax.text(mooring['lon'][0]+3/60, mooring['lat'][0]+.15/60, 'Mooring', fontsize=6, transform=ccrs.PlateCarree(), zorder=3)

    #ax.legend(loc='upper right', fontsize=6, frameon=True)

    if savefig:
        plt.savefig(__figdir__+'SLA'+str(tind)+'.'+plotfiletype,**savefig_args)

    return ax

# %%
def add_vel_quiver(tind,ax=plt.gca()):
    if ax is None:
        ax = plt.gca()

    u = np.squeeze(ds.ugos.isel(time=tind)) #dtype=object
    v = np.squeeze(ds.vgos.isel(time=tind))
    skip = 5
    scalefac = 10
    q = ax.quiver(ds.longitude.values[::skip], ds.latitude.values[::skip], u.values[::skip,::skip], v.values[::skip,::skip], scale=scalefac, transform=ccrs.PlateCarree())
    x0 = mooring['lon'][0] + dx + 2
    y0 = mooring['lat'][0] + dy + 1.5
    ax.quiverkey(q,x0,y0,0.25, '0.25 m/s', zorder=3, transform=ccrs.PlateCarree())
    #ax.quiver(np.array([x0]), np.array([y0]), -np.array([0.25/np.sqrt(2)],), np.array([0.25/np.sqrt(2)]), scale=scalefac, transform=ccrs.PlateCarree(),zorder=3)
    #ax.text(x0+3/60, y0+.15/60, '0.25 m/s', fontsize=6,transform=ccrs.PlateCarree())




# %%
ds

# %%
fig = plt.figure()
tind=-1
ax = plot_SSH_map(tind)

# %%
add_vel_quiver(tind, ax=ax)

# %%
# Make a movie of the SSH
if domovie:
    fig = plt.figure()
    for tind in range(len(ds.time)):
        ax = plot_SSH_map(tind)
        add_vel_quiver(tind, ax=ax)


# %%

# %%
# !ffmpeg -i SLA%d.png -r 10 SSH_April_20.avi



# %%
