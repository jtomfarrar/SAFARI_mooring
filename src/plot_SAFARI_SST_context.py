# Plot OISST for the Bay of Bengal (EKAMSAT 2025 cruise)
#

# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs                   # import projections
import cartopy
import geopandas as gpd # for the EEZ shapefile
#import functions
import pandas as pd

# %%
# Change to this directory
home_dir = os.path.expanduser("~")

# To work for Tom and other people
if os.path.exists(home_dir + '/Python/SAFARI_mooring/src'):
    os.chdir(home_dir + '/Python/SAFARI_mooring/src')

# %%
# %matplotlib inline
%matplotlib widget
# %matplotlib qt5
plt.rcParams['figure.figsize'] = (6,6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400

savefig = True # set to true to save plots as file

__figdir__ = '../img/SST_movie/'
sst_figdir = '../img/SST_movie/'
os.system('mkdir  -p ' + __figdir__) #make directory if it doesn't exist
savefig_args = {'bbox_inches':'tight', 'pad_inches':0.2}
kml_savefig_args = {'bbox_inches':'tight', 'pad_inches':0, 'transparent':True}
plotfiletype='png'

# %%
# clear the directory
#os.system('rm -f ' + __figdir__ + '*')


# %%
# Download the data if needed
# https://psl.noaa.gov/thredds/catalog/Datasets/noaa.oisst.v2.highres/catalog.html?dataset=Datasets/noaa.oisst.v2.highres/sst.day.mean.2023.nc

# download the data to ../data/external if it does not already exist there
data_dir = '../data/external'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

#url = 'https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.oisst.v2.highres/sst.day.mean.2023.nc'
url = 'https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.oisst.v2.highres/sst.day.mean.2026.nc'


filename = os.path.join(data_dir, os.path.basename(url))
# Commenting out since cruise is over:

if not os.path.exists(filename):
    import time
    max_retries = 3
    
    # Try using requests library for more robust downloads
    try:
        import requests
        use_requests = True
    except ImportError:
        use_requests = False
        import urllib.request
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading {url} (attempt {attempt + 1}/{max_retries})...")
            
            if use_requests:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filename, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                pct = (downloaded / total_size) * 100
                                print(f"  Downloaded: {downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB ({pct:.1f}%)", end='\r')
                print("\nDownload successful!")
            else:
                urllib.request.urlretrieve(url, filename)
                print("Download successful!")
            break
            
        except Exception as e:
            print(f"\nDownload failed: {e}")
            if os.path.exists(filename):
                os.remove(filename)
            if attempt < max_retries - 1:
                print(f"Retrying in 10 seconds...")
                time.sleep(10)
            else:
                print("Max retries reached. File may be temporarily unavailable or network is unstable.")
                raise


'''if not os.path.exists(filename):
    import urllib.request
    urllib.request.urlretrieve(url, filename)'''
'''else: # check if the file is older than 0.5 days
    # Get the last modified time of the file
    import datetime
    import time
    import urllib.request
    # Get the last modified time of the file
    last_modified_time = os.path.getmtime(filename)
    # Convert the last modified time to a human-readable format
    last_modified_time = datetime.datetime.fromtimestamp(last_modified_time)
    print("Last modified time:", last_modified_time)
    print("Time now:", datetime.datetime.now())
    # compute age of file:
    age = datetime.datetime.now() - last_modified_time
    print("Age of file:", age)
    # If the file is older than 0.5 days, download it again
    if age.total_seconds() > 43200:  # 0.5 days in seconds
        print("File is older than 0.5 days, downloading again")
        urllib.request.urlretrieve(url, filename)'''



# %%
# Nominal mooring location: 
mooring = dict(
    lon = [-158],
    lat = [33.4])
mooring['lon'][0] = (mooring['lon'][0] + 360) % 360


# %%
# Convert waypoints from decimal degrees to degrees and decimal minutes
'''def decimal_to_dms(value, direction_positive, direction_negative):
    degrees = int(value)
    minutes = (value - degrees) * 60
    direction = direction_positive if value >= 0 else direction_negative
    return f"{degrees}° {minutes:.2f}' {direction}"
# Current first survey waypoint
# OLD: wpt = dict(lon=[86 + 34.62/60], lat=[13 + 48.03/60])
#  13.875100°   86.482300°
#  13.323279°   86.501382°
#  Not doing: 14.466317°   88.833707°  
#  12.80°   86.50°
#  12.30°   86.50°
#  12.00°   88.50°

#  13.172930°   87.664760°
#  13.170381°   87.510335°
#  13.315435°   87.510009°
#  13.313899°   87.663535°
# make a lon/lat dictionary of the waypoints above, rounding to 4 decimal places
# First square aroung WGs on May 8 midnight -8am LT:
# uctd_wpt = dict(lon=[87.6648, 87.5103, 87.5100, 87.6635], lat=[13.1729, 13.1704, 13.3154, 13.3139])
# For May 8, 1000-1800 LT
#  13.276094°   87.529407°
#  13.275738°   87.677471°
#  13.132261°   87.676757°
#  13.132260°   87.530091°
uctd_wpt = dict(lon=[87.5294, 87.6775, 87.6768, 87.5301], lat=[13.2761, 13.2757, 13.1323, 13.1323]) 
# G:\Shared drives\AirSeaLab_Shared\ASTRAL_2025\PAYLOAD\MAT
# IDA.PLD2_TAB1['lon']

# Drifter waypoints for May 9
# 86°E, 12°N (DWSD and SG181)\\
# 86°E 11.5°N (DWSBWD)\\
# 86°E, 11°N (DWSD)\\
# 86.5°E, 11°N (DWSD)\\
# 86.5°E, 11.5°N (DWSD)\\
# 86.5°E, 12°N (DWSD)\\

wpt_drifters_May9 = dict(lon=[86, 86, 86, 86.5, 86.5, 86.5], lat=[12, 11.5, 11, 11, 11.5, 12])


CTD_waypoint= dict(lon=[87.511854], lat=[13.233402])'''

'''# Add new waypoints to wpt
wpt = dict(lon=[88.83, 88.83, 86.48, 86.50, 86.50, 86.50, 87.5, 88.5], lat=[12.33, 15, 13.87, 13.32, 12.80, 12.30, 13.25, 12.0])
# add uctd waypoints to wpt
wpt['lon'] += uctd_wpt['lon']
wpt['lat'] += uctd_wpt['lat']
# add drifter waypoints to wpt
wpt['lon'] += wpt_drifters_May9['lon']
wpt['lat'] += wpt_drifters_May9['lat']
# WG recovery waypoint
wpt['lon'] += [87.67]
wpt['lat'] += [13.25]


wpt_dms = [
    f"{decimal_to_dms(lat, 'N', 'S')}, {decimal_to_dms(lon, 'E', 'W')}"
    for lat, lon in zip(wpt['lat'], wpt['lon'])
]

print("Waypoints in DMS format (lat, lon):")
for waypoint in wpt_dms:
    print(waypoint)

wpt_uctd_dms = [
    f"{decimal_to_dms(lat, 'N', 'S')}, {decimal_to_dms(lon, 'E', 'W')}"
    for lat, lon in zip(uctd_wpt['lat'], uctd_wpt['lon'])
]

print("UCTD waypoints in DMS format (lat, lon):")
for waypoint in wpt_uctd_dms:
    print(waypoint)

wpt_CTD_dms = [
    f"{decimal_to_dms(lat, 'N', 'S')}, {decimal_to_dms(lon, 'E', 'W')}"
    for lat, lon in zip(CTD_waypoint['lat'], CTD_waypoint['lon'])
]
print("CTD waypoints in DMS format (lat, lon):")
for waypoint in wpt_CTD_dms:
    print(waypoint)

wpt_drifters_May9_dms = [
    f"{decimal_to_dms(lat, 'N', 'S')}, {decimal_to_dms(lon, 'E', 'W')}"
    for lat, lon in zip(wpt_drifters_May9['lat'], wpt_drifters_May9['lon'])
]
print("Drifter waypoints in DMS format (lat, lon):")
for waypoint in wpt_drifters_May9_dms:
    print(waypoint)'''

# %%

'''
# Get ship position from this url:
# https://www.ocean.washington.edu/files/thompson.txt
ship_pos_url = 'https://www.ocean.washington.edu/files/thompson.txt'
ship_pos = pd.read_csv(ship_pos_url, sep=',', header=None, names=['asset', 'DD-MM-YYYY', 'HH:MM:SS', 'lat', 'lon', 'T', 'NULL1', 'NULL2', 'BPR', 'NULL3', 'NULL4', 'NULL5', 'NULL6', 'NULL7', 'NULL8', 'NULL9', 'Voyage'])
print(ship_pos.head())
latitudes = ship_pos['lat']
longitudes = ship_pos['lon']'''
# %%
# Load the data
ds = xr.open_dataset(filename)

# %% Shift ds to have longitude from -180 to 180
ds = ds.assign_coords(lon=(ds.lon + 180) % 360 - 180)
ds = ds.sortby(ds.lon)


# %%
zoom = True
domovie = False
if zoom:
    dx = 7.5
    dy = 7.5
    xmin, xmax = (mooring['lon'][0] - dx, mooring['lon'][0] + dx)
    ymin, ymax = (mooring['lat'][0] - dy, mooring['lat'][0] + dy)
else:
    dx = 30
    dy = 30
    xmin, xmax = (mooring['lon'][0] - dx, mooring['lon'][0] + dx)
    ymin, ymax = (mooring['lat'][0] - dy, mooring['lat'][0] + dy)
# %%
ds_ssh = xr.open_dataset('../data/external/aviso.nc')

# %%
# Reverted the add_vel_quiver function to its original state

def add_vel_quiver(tind, ax=plt.gca()):
    if ax is None:
        ax = plt.gca()

    u = np.squeeze(ds_ssh.ugos.isel(time=tind))  # dtype=object
    v = np.squeeze(ds_ssh.vgos.isel(time=tind))
    skip = 3
    scalefac = 10
    q = ax.quiver(ds_ssh.longitude.values[::skip], ds_ssh.latitude.values[::skip], u.values[::skip, ::skip], v.values[::skip, ::skip], scale=scalefac, transform=ccrs.PlateCarree())
    x0 = 81.5
    y0 = 17.33
    ax.quiverkey(q, x0, y0, 0.25, '0.25 m/s', zorder=5, transform=ccrs.PlateCarree())


######################

# %%
## Plot the data
def plot_map(data, levels, title='', outfile='', savefig=False, ax=None):
    if title is None:
        title = ''
    if outfile is None:
        outfile = title
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})  # Create a new axis if none is provided
    coast = cartopy.feature.GSHHSFeature(scale="full")
    ax.add_feature(coast, zorder=3, facecolor=[.6, .6, .6], edgecolor='black')

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Add country boundaries
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5, zorder=10)
    ax.add_feature(cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='50m', facecolor='none'), zorder=10)
    ax.add_feature(cartopy.feature.RIVERS, edgecolor='blue', zorder=10, alpha=0.25)

    # Plot the data
    cs = ax.pcolormesh(data.lon, data.lat, data, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree())
    cb = plt.colorbar(cs, ax=ax, fraction=0.022, extend='both')
    cb.set_label('SST [$\\circ$C]', fontsize=10)
    ax.axis('scaled')
    ax.set_title(title)

    # Add 2024 site
    site_2024 = ax.plot(mooring['lon'], mooring['lat'], 'o', color='k', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='2024 site')

    if savefig:
        outfile2 = outfile.replace(' ', '_')
        plt.savefig(__figdir__ + outfile2 + '.' + plotfiletype, **savefig_args)
    return ax, site_2024




# %%
# Make one plot for the last time in the file
t1 = ds.time[-1].values
fstr = t1.astype('datetime64[D]')
t1 = str(fstr)
sst = ds.sst.sel(lon=slice(xmin,xmax), lat=slice(ymin,ymax),time=t1)



# %%
BD = ['BD08', 'BD09', 'BD10', 'BD13', 'RAMA', 'RAMA']
pts_lon = [89.175, 89.124, 87.991, 87.00, 89.04, 88.51]
pts_lat = [17.817, 17.460, 16.322, 13.99, 15.04, 12.01]

# %%
# Add EEZ
# Download the file to ../data/external/
# https://www.marineregions.org/downloads.php#eez

EEZ_file = '../data/external/World_EEZ_v12_20231025.zip'
if not os.path.isfile(EEZ_file):
    import urllib.request
    url = 'https://www.marineregions.org/downloads.php#eez'
    urllib.request.urlretrieve(url, EEZ_file)
# %%
# Unzip the file
if not os.path.exists(data_dir + '/World_EEZ_v12_20231025'):
    import zipfile
    # Unzip the file
    data_dir = '../data/external'
    EEZ_file = data_dir + '/World_EEZ_v12_20231025.zip'
    with zipfile.ZipFile(EEZ_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)


# Read the shapefile
shapefile = data_dir + '/World_EEZ_v12_20231025/eez_v12.shp'
gdf = gpd.read_file(shapefile)

# %%
# Plot the shapefile on the map
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})  # Define ax with cartopy projection
plt.set_cmap(cmap=plt.get_cmap('turbo'))
levels = np.arange(29.5, 32, 0.25)

# Plot the SST map on the same axis
ax, site_2024 = plot_map(sst, levels, title='SST,' + t1, savefig=False, ax=ax)

# Add the NRL target point
# nrl_point = ax.plot(NRL_target['lon'], NRL_target['lat'], 'o', color='g', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='NRL target')

# Plot the BD/RAMA moorings
bd = ax.plot(pts_lon, pts_lat, 'o', color='m', markeredgecolor='k', markersize=8, transform=ccrs.PlateCarree(), label='BD/RAMA moorings')
for i in range(len(pts_lon)):
    # label the points
    ax.text(pts_lon[i] + 0.1, pts_lat[i] + 0.1, BD[i], fontsize=12, color='c', transform=ccrs.PlateCarree(), zorder=4)


# Plot the shapefile
gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2, alpha=0.9, transform=ccrs.PlateCarree(), zorder=2)

ax.grid(True)

# Add titles and labels
ax.set_title('SST,' + t1)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

# Move the first X values of wpt to a variable called wpt_old
X=18
wpt_old = dict(lon=wpt['lon'][:X], lat=wpt['lat'][:X])
wpt = dict(lon=wpt['lon'][X:], lat=wpt['lat'][X:])

# Update the plotting to include old_wpt points in grey with a different legend entry
old_wpt = plt.plot(wpt_old['lon'], wpt_old['lat'], 'o', color='grey', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='Old waypoints')
next_wpt = plt.plot(wpt['lon'], wpt['lat'], 'o', color='r', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='Current waypoints')

tgt = plt.plot(longitudes, latitudes, 'o', color='b', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='Ship Position')
plt.legend([site_2024[0], bd[0], next_wpt[0], old_wpt[0], tgt[0]], ['2024 site', 'BD/RAMA moorings', 'Current waypoints', 'Old waypoints', 'Ship Position'], loc='upper right', framealpha=0.8)

#plt.legend(framealpha=0.8)
plt.show()


tind=-1
add_vel_quiver(tind, ax=ax)

if savefig:
    outfile2 = 'SST_UV_map_' + t1.replace(' ', '_')
    plt.savefig(__figdir__ + outfile2 + '.' + plotfiletype, **savefig_args)


# %%
# Make a plot for a KML file:
# Do a version of the plot where the axes take the whole figure
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
plt.set_cmap(cmap=plt.get_cmap('turbo'))
cs = ax.pcolormesh(sst.lon, sst.lat, sst, vmin=levels[0], vmax=levels[-1], transform=ccrs.PlateCarree())
add_vel_quiver(tind, ax=ax)
#gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2, alpha=0.9, transform=ccrs.PlateCarree(), zorder=2)

# remove all whitespace
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# Get axis limits to restore after plotting EEZ
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
# Add EEZ
gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2, alpha=0.9, transform=ccrs.PlateCarree(), zorder=2)
# restore axis limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
if savefig:
    outfile2 = 'KML_SST_UV_map_' + t1.replace(' ', '_')
    plt.savefig(__figdir__ + outfile2 + '.' + plotfiletype, **kml_savefig_args)

# %%
functions.create_kml_file(kml_name=__figdir__ + outfile2, overlay_name='SST_UV', plot_file=outfile2 + '.' + plotfiletype, pts_lon=pts_lon, pts_lat=pts_lat, BD=BD, mooring=mooring, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# %%
# Calculate the SST difference over the last week
# Get the time range for the last week
t2 = (ds.time[-7].values).astype('datetime64[D]').astype(str)
t1 = (ds.time[-1].values).astype('datetime64[D]').astype(str)
sst_2 = ds.sst.sel(time=t1).sel(lon=slice(xmin, xmax), lat=slice(ymin, ymax))
sst_1 = ds.sst.sel(time=t2).sel(lon=slice(xmin, xmax), lat=slice(ymin, ymax))
sst_diff = sst_2 - sst_1

# Plot the SST difference
levels_diff = np.linspace(-1, 1, 21)  # Adjust levels as needed
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
plt.set_cmap(cmap=plt.get_cmap('RdBu_r'))  # Diverging colormap for differences

cs = ax.pcolormesh(sst_diff.lon, sst_diff.lat, sst_diff, vmin=levels_diff[0], vmax=levels_diff[-1], transform=ccrs.PlateCarree())
cb = plt.colorbar(cs, ax=ax, fraction=0.022, extend='both')
cb.set_label('SST Difference (°C)', fontsize=10)

# Add map features
coast = cartopy.feature.GSHHSFeature(scale="full")
ax.add_feature(coast, zorder=3, facecolor=[.6, .6, .6], edgecolor='black')
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5, zorder=10)
ax.add_feature(cartopy.feature.RIVERS, edgecolor='blue', alpha=0.25, zorder=10)

# Add gridlines
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Add the NRL target point
# nrl_point = ax.plot(NRL_target['lon'], NRL_target['lat'], 'o', color='g', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='NRL target')

# Add titles and labels
ax.set_title(f'SST Difference: {t1} minus {t2}')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
# add the eez boundaries
gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2, alpha=0.9, transform=ccrs.PlateCarree(), zorder=2)
# Add BD/RAMA moorings
bd = ax.plot(pts_lon, pts_lat, 'o', color='m', markeredgecolor='k', markersize=8, transform=ccrs.PlateCarree(), label='BD/RAMA moorings')
for i in range(len(pts_lon)):
    # label the points
    ax.text(pts_lon[i] + 0.1, pts_lat[i] + 0.1, BD[i], fontsize=12, color='m', transform=ccrs.PlateCarree(), zorder=4)

site_2024 = ax.plot(mooring['lon'], mooring['lat'], 'o', color='k', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='2024 site')

add_vel_quiver(tind, ax=ax)
# Plot the Current waypoints and ship
old_wpt = plt.plot(wpt_old['lon'], wpt_old['lat'], 'o', color='grey', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='Old waypoints')
next_wpt = plt.plot(wpt['lon'], wpt['lat'], 'o', color='r', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='Current waypoints')
tgt = plt.plot(longitudes, latitudes, 'o', color='b', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='Ship Position')
plt.legend([site_2024[0], bd[0], next_wpt[0], old_wpt[0], tgt[0]], ['2024 site', 'BD/RAMA moorings', 'Current waypoints', 'Old waypoints', 'Ship Position'], loc='upper right', framealpha=0.8)


plt.show()

if savefig:
    outfile2 = 'Delta_SST_UV_map_' + t1.replace(' ', '_')
    plt.savefig(__figdir__ + outfile2 + '.' + plotfiletype, **savefig_args)

# %%
# Now make a plot for a KML file:
# Do a version of the plot where the axes take the whole figure
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
plt.set_cmap(cmap=plt.get_cmap('RdBu_r'))  # Diverging colormap for differences
cs = ax.pcolormesh(sst_diff.lon, sst_diff.lat, sst_diff, vmin=levels_diff[0], vmax=levels_diff[-1], transform=ccrs.PlateCarree())
add_vel_quiver(tind, ax=ax)
#gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2, alpha=0.9, transform=ccrs.PlateCarree(), zorder=2)

# remove all whitespace
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# Get axis limits to restore after plotting EEZ
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
# Add EEZ
gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2, alpha=0.9, transform=ccrs.PlateCarree(), zorder=2)
# restore axis limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
if savefig:
    outfile2 = 'KML_Delta_SST_UV_map_' + t1.replace(' ', '_')
    plt.savefig(__figdir__ + outfile2 + '.' + plotfiletype, **kml_savefig_args)

# %%
functions.create_kml_file(kml_name=__figdir__ + outfile2, overlay_name='Delta_SST_UV', plot_file=outfile2 + '.' + plotfiletype, pts_lon=pts_lon, pts_lat=pts_lat, BD=BD, mooring=mooring, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


# %%

# Plot the Current waypoints and ship
old_wpt = plt.plot(wpt_old['lon'], wpt_old['lat'], 'o', color='grey', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='Old waypoints')
next_wpt = plt.plot(wpt['lon'], wpt['lat'], 'o', color='r', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='Current waypoints')
tgt = plt.plot(longitudes, latitudes, 'o', color='b', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='Ship Position')
plt.legend([site_2024, bd[0], next_wpt[0], old_wpt[0], tgt[0]], ['2024 site', 'BD/RAMA moorings', 'Current waypoints', 'Old waypoints', 'Ship Position'], loc='upper right', framealpha=0.8)







# %%
# Export the Current waypoints to a KML file; this differs from the other function 
# because it only does the waypoints

# %%
# Create a KML file for the Current waypoints and NRL target
functions.waypoints_to_kml(kml_name=__figdir__ + 'current_waypoint', wpt=wpt)
functions.waypoints_to_kml(kml_name=__figdir__ + 'old_waypoint', wpt=wpt_old)
functions.waypoints_to_kml(kml_name=__figdir__ + 'uctd_waypoint', wpt=uctd_wpt)
functions.waypoints_to_kml(kml_name=__figdir__ + 'drifter_waypoint', wpt=wpt_drifters_May9)
# Add the NRL target to the KML file
#functions.points_to_kml(kml_name=__figdir__ + 'NRL_target', wpt=NRL_target,pt_label='NRL target')


# %%
# Make a plot of current speed and direction
# Plot the current speed and direction

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
plt.set_cmap(cmap=plt.get_cmap('turbo'))
# Plot the current speed
u = np.squeeze(ds_ssh.ugos.isel(time=tind)) #dtype=object
v = np.squeeze(ds_ssh.vgos.isel(time=tind))
# Calculate the speed
speed = np.sqrt(u**2 + v**2)
# Plot the speed
cs = ax.pcolormesh(ds_ssh.longitude, ds_ssh.latitude, speed, vmin=0, vmax=0.75, transform=ccrs.PlateCarree())
cb = plt.colorbar(cs, ax=ax, fraction=0.022, extend='both')
cb.set_label('Current Speed [m/s]', fontsize=10)
# Add map features
coast = cartopy.feature.GSHHSFeature(scale="full")
ax.add_feature(coast, zorder=3, facecolor=[.6, .6, .6], edgecolor='black')
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=0.5, zorder=10)
ax.add_feature(cartopy.feature.RIVERS, edgecolor='blue', alpha=0.25, zorder=10)
# Add gridlines
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
# Add titles and labels
ax.set_title('Current Speed')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

add_vel_quiver(tind, ax=ax)
# Add the BD/RAMA moorings
bd = ax.plot(pts_lon, pts_lat, 'o', color='m', markeredgecolor='k', markersize=8, transform=ccrs.PlateCarree(), label='BD/RAMA moorings')
for i in range(len(pts_lon)):
    # label the points
    ax.text(pts_lon[i] + 0.1, pts_lat[i] + 0.1, BD[i], fontsize=12, color='w', transform=ccrs.PlateCarree(), zorder=4)
# Add the 2024 site
site_2024 = ax.plot(mooring['lon'], mooring['lat'], 'o', color='k', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='2024 site')
# nrl_point = ax.plot(NRL_target['lon'], NRL_target['lat'], 'o', color='g', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='NRL target')

# Add the EEZ boundaries
gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2, alpha=0.9, transform=ccrs.PlateCarree(), zorder=2)
# Add the Current waypoints and ship
old_wpt = plt.plot(wpt_old['lon'], wpt_old['lat'], 'o', color='grey', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='Old waypoints')
next_wpt = plt.plot(wpt['lon'], wpt['lat'], 'o', color='r', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='Current waypoints')
tgt = plt.plot(longitudes, latitudes, 'o', color='b', markeredgecolor='w', markersize=8, transform=ccrs.PlateCarree(), label='Ship Position')
plt.legend([site_2024[0], bd[0], next_wpt[0], old_wpt[0], tgt[0]], ['2024 site', 'BD/RAMA moorings', 'Current waypoints', 'Old waypoints', 'Ship Position'], loc='upper right', framealpha=0.8)
plt.show()
if savefig:
    outfile2 = 'Current_Speed_map_' + t1.replace(' ', '_')
    plt.savefig(__figdir__ + outfile2 + '.' + plotfiletype, **savefig_args)
# %%
# Make a plot for a KML file:
# Do a version of the plot where the axes take the whole figure
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
plt.set_cmap(cmap=plt.get_cmap('turbo'))
# Plot the current speed
cs = ax.pcolormesh(ds_ssh.longitude, ds_ssh.latitude, speed, vmin=0, vmax=0.75, transform=ccrs.PlateCarree())
add_vel_quiver(tind, ax=ax)
# remove all whitespace
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# Get axis limits to restore after plotting EEZ
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
# Add EEZ
gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2, alpha=0.9, transform=ccrs.PlateCarree(), zorder=2)
# restore axis limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
if savefig:
    outfile2 = 'KML_Current_Speed_map_' + t1.replace(' ', '_')
    plt.savefig(__figdir__ + outfile2 + '.' + plotfiletype, **kml_savefig_args)
# %%
functions.create_kml_file(kml_name=__figdir__ + outfile2, overlay_name='Current_Speed', plot_file=outfile2 + '.' + plotfiletype, pts_lon=pts_lon, pts_lat=pts_lat, BD=BD, mooring=mooring, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

# %%
'''
import socket
import simplekml

# UDP server configuration
UDP_IP = "172.26.4.188"  # Listen on all interfaces
UDP_PORT = 55555

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Create a KML object
kml = simplekml.Kml()

# Loop to receive data
while True:
    data, addr = sock.recvfrom(1024)
    # Parse the data (replace with your parsing logic)
    latitude = float(data.decode("utf-8").split(",")[0])
    longitude = float(data.decode("utf-8").split(",")[1])
    # Add a Placemark in KML
    pnt = kml.newpoint(name=f"Location {latitude}, {longitude}", coords=[(longitude, latitude)])
    # Save the KML
    kml.save("location_data.kml")
    print(f"Received location: {latitude}, {longitude}")
    time.sleep(1)
'''
# %%
