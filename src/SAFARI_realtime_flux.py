
# 2025/11/19 Benjamin Greenwood compute fluxes using COARE 3.6 bulk flux algorithm
# 2026/02/07 Tom Farrar add interpolation of nans in input variables and plotting of input variables;
# added more plots of output variables; added saving output to netCDF file with metadata

# %%
import os
# %%
# Change to this directory
home_dir = os.path.expanduser("~")

# To work for Tom and other people; other users can add elif statements for their own directory structure
if os.path.exists(home_dir + '/Python/SAFARI_mooring/src'):
    os.chdir(home_dir + '/Python/SAFARI_mooring/src')

# %%
from coare36vn_zrf_et import coare36vn_zrf_et
import gsw
import datetime
import json
import numpy as np
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import ssl
import xarray as xr
from coare36_variables_config import get_variables_info
import matplotlib.pyplot as plt
# %%
%matplotlib widget
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 400

# %%
def _ssl_context():
  # Prefer certifi bundle if available to avoid system CA issues.
  try:
    import certifi  # type: ignore
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

ASIMET = load_json_url('https://uop.whoi.edu/currentprojects/SAFARI/data/SAFARI_ASIMET.json', verify_ssl=False)
CR6 = load_json_url('https://uop.whoi.edu/currentprojects/SAFARI/data/SAFARI_buoy.json', verify_ssl=False)

# %%
t0 = datetime.datetime(2025,11,21) # SAFARI buoy deployed from R/V Sikuliaq at this time

# get timestamps for Campbell and ASIMET datasets
ctime = np.array([ datetime.datetime.fromtimestamp(t).replace(minute=0,second=0,microsecond=0) for t in CR6['MET']['time'] ])
atime = np.array([ datetime.datetime.fromtimestamp(t).replace(minute=0,second=0,microsecond=0) for t in ASIMET['ASIMET']['time'] ])
wave_time = np.array([ datetime.datetime.fromtimestamp(t).replace(minute=0,second=0,microsecond=0) for t in CR6['FF']['time'] ])

# define preferred sensors
time = np.arange(t0,datetime.datetime.now(),datetime.timedelta(hours=1)).astype(datetime.datetime)

# %%
def select_sensor(time,primary_time,primary_var,secondary_time = None,secondary_var = None):
  var = np.full(len(time),np.nan) # initialize variable to NaN
  if not secondary_time is None:
    i1 = np.where(np.in1d(time,secondary_time))[0]
    i2 = np.where(np.in1d(secondary_time,time))[0]
    var[i1] = np.array(secondary_var)[i2]
  i1 = np.where(np.in1d(time,primary_time))[0]
  i2 = np.where(np.in1d(primary_time,time))[0]
  var[i1] = np.array(primary_var)[i2]
  return var

#def rain_rate(r,scale=None):
#  r = np.array(r)
#  r[r == None] = 0
#  if scale:
#    r = r / scale
#  dr = np.diff(r)
#  dr = np.insert(dr,0,0)
#  thresh = 0.02
#  dr[dr>50] = 0
#  dr[dr<thresh] = 0 # set all negatvie precip values and values less than threshold to zero
#  return dr

def rain_rate(r):
  r = np.array(r)
  r[r == None] = 0
  dr = np.diff(r)
  dr = np.insert(dr,0,0)
  # remove precip below user defined threshold
  thresh = 4
  dr[ (dr < thresh) & (dr > -500)] = 0

  # identify siphoning events and account for them
  siphons = np.where(dr < -500)[0]
  s = 0
  while s < len(siphons):
    i = siphons[s]
    # drain recorded over two averaging periods
    if dr[i+1] < 0:
      dr[i] = max( 0, 5000 - r[i-1]) # replace first hour with 50mm - last measurement prior to drain
      dr[i+1] = r[i+1] # replace 2nd hour with lowest measurement after drain
      s = s + 2
    # drain recorded over a single averaging period
    else:
      dr[i] = max(r[i],dr[i] + 5000) # dr[i]+5000 can be negative if filled value > 5000
      s = s + 1
  # Divide dr by dt to get rain rate in mm/hr; dt is 1 hour in this case, so dr is already in mm/hr

  return dr
# %%
U    = select_sensor(time,atime,ASIMET['ASIMET']['wspd'])
zu   = 3.4
t    = select_sensor(time,ctime,CR6['MET']['HC2_temperature'],atime,ASIMET['ASIMET']['atmp'])
zt   = 3.03 # HC2A mounted at 2.28 + deck (0.75), HRH229 mounted at 2.27 + deck
rh   = select_sensor(time,ctime,CR6['MET']['HC2_humidity'],atime,ASIMET['ASIMET']['rh'])
zq   = 3.03 # HC2A mounted at 2.28 + deck (0.75), HRH229 mounted at 2.27 + deck
P    = select_sensor(time,atime,ASIMET['ASIMET']['bpr'])
ts   = select_sensor(time,ctime,CR6['MET']['SBE37_temp'],atime,ASIMET['ASIMET']['stmp'])
sw   = select_sensor(time,ctime,CR6['MET']['SR30_flux'],atime,ASIMET['ASIMET']['swr'])
lw   = select_sensor(time,atime,ASIMET['ASIMET']['lwr'])
lat  = select_sensor(time,ctime,CR6['MET']['latitude'])
lon  = select_sensor(time,ctime,CR6['MET']['longitude'])
cond = select_sensor(time,ctime,CR6['MET']['SBE37_cond'],atime,ASIMET['ASIMET']['cond'])
stmp = select_sensor(time,ctime,CR6['MET']['SBE37_temp'],atime,ASIMET['ASIMET']['stmp'])
jd   = 0
zi   = 600
rain = select_sensor(time,atime,rain_rate(np.array(ASIMET['ASIMET']['prc'])))
with np.printoptions(threshold=np.inf):
  print(rain)
Ss   = gsw.SP_from_C( cond, stmp, 0 )
zrfu = 10.0 # Reference height for wind [m]
zrft = 3 # Reference height for temperature [m]
zrfq = 3 # Reference height for humidity [m]

# %%
# Before estimating fluxes, check inputs for nans
input_vars = [U, t, rh, P, ts, sw, lw, lat, lon, cond, stmp, rain, Ss]
input_var_names = ['U', 't', 'rh', 'P', 'ts', 'sw', 'lw', 'lat', 'lon', 'cond', 'stmp', 'rain', 'Ss']

# %%
# Plot all input variables to check for any obvious issues
fig, axes = plt.subplots(len(input_vars), 1, figsize=(12, 20), sharex=True)
for var, name, ax in zip(input_vars, input_var_names, axes):
  ax.plot(time, var)
  ax.set_title(name)
  ax.grid()
plt.xlabel('Time')
plt.tight_layout()
plt.suptitle('SAFARI Mooring Input Variables')
plt.show()

# %%
# Now interpolate any nans in the input variables before passing to coare36vn_zrf_et
# First store a copy of the original input variables with nans for later comparison; just copy input_vars and input_var_names to original_input_vars and original_input_var_names
original_input_vars = input_vars.copy()
original_input_var_names = input_var_names.copy()

# Convert datetime to numeric values for interpolation
time_numeric = np.arange(len(time))

# Interpolate nans in each variable
for i, var in enumerate(input_vars):
  if np.isnan(var).any():
    valid_idx = ~np.isnan(var)
    var_interp = np.interp(time_numeric, time_numeric[valid_idx], var[valid_idx])
    input_vars[i] = var_interp
    print(f"Interpolated nans in {input_var_names[i]}")
  
# %%
# Plot all input variables again to check that nans have been interpolated
fig, axes = plt.subplots(len(input_vars), 1, figsize=(12, 20), sharex=True)
for var, name, ax in zip(input_vars, input_var_names, axes):
  ax.plot(time, var, label='Interpolated')
  # Also plot original variable with nans for comparison
  original_var = original_input_vars[input_var_names.index(name)]
  ax.plot(time, original_var, label='Original', alpha=0.85)
  ax.set_title(name)
  ax.grid()
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.suptitle('SAFARI Mooring Input Variables (Interpolated)')
plt.show()

# %%
# replace input variables with interpolated versions for flux calculation
U, t, rh, P, ts, sw, lw, lat, lon, cond, stmp, rain, Ss = input_vars
# %%
#outputs = coare36vn_zrf_et(U,zu,t,zt,rh,zq,P,ts,sw,lw,lat,lon,jd,zi,rain,Ss,cp=None,sigH=None,zrf_u=zrfu,zrf_t=zrft,zrf_q=zrfq)
outputs = coare36vn_zrf_et(U,zu,t,zt,rh,zq,P,ts,sw,lw,lat,lon,jd,zi,rain,Ss,cp=None,sigH=None,zrf_u=zrfu,zrf_t=zrft,zrf_q=zrfq, coolskin=True, albedo='constant', return_albedo=False, wind_direction=None)

print(outputs)

# %%
# Use coare36_variables_config.py to match up with each field in the output ndarray
variables_info = get_variables_info(albedo_method='constant')
output_var_names = list(variables_info.keys())
# Drop the last 4 variables which are not present in the output of coare36vn_zrf_et
output_var_names = output_var_names[:outputs.shape[1]]

if outputs.shape[1] != len(output_var_names):
  raise ValueError(f"Expected {len(output_var_names)} outputs, got {outputs.shape[1]}")

ds = xr.Dataset(coords={"time": time})
for i, name in enumerate(output_var_names):
  data = outputs[:, i]
  meta = variables_info.get(name, {})
  if meta.get("reverse_sign"):
    data = -1 * data
  da = xr.DataArray(data, dims=("time",), coords={"time": time})
  for k, v in meta.items():
    if k != "reverse_sign":
      da.attrs[k] = v
  ds[name] = da

out_path = os.path.join(home_dir, "Python/SAFARI_mooring/data/SAFARI_fluxes.nc")
ds.to_netcdf(out_path)

# %%
ds
# %%
# Plot some variables
plt.figure()
ds['latent_heat_flux'].plot(label='Latent Heat Flux')
ds['sensible_heat_flux'].plot(label='Sensible Heat Flux')
plt.legend()
plt.title('SAFARI Mooring Fluxes')
# autoformatted x-axis with dates
#plt.gcf().autofmt_xdate()
plt.ylabel('Flux (W/m^2)')
plt.grid()
plt.show()

# %%
plt.figure()
ds['wind_stress_eastward'].plot(label='Wind Stress X')
ds['wind_stress_northward'].plot(label='Wind Stress Y')
plt.legend()
plt.title('SAFARI Mooring Wind Stress')
plt.ylabel('Wind Stress (N/m^2)')
plt.grid()
plt.show()
# %%
fig, axes = plt.subplots(6, 1, figsize=(8, 8), sharex=True)
plt.subplot(6,1,1)
ds['wind_speed_at_reference_height'].plot(label='10m wind speed')
plt.xlabel('')
plt.ylabel('[m/s]')
plt.subplot(6,1,2)
plt.plot(ds.time,ts,label='SST')
ds['air_temperature_at_reference_height'].plot(label='Air Temp at '+str(zrft)+'m')
plt.legend()
plt.xlabel('')
plt.ylabel('[$^\circ$C]')
plt.subplot(6,1,3)
ds['relative_humidity_at_reference_height'].plot(label='Humidity at '+str(zrfq)+'m')
plt.xlabel('')
plt.ylabel('[%]')
plt.legend()
plt.subplot(6,1,4)
plt.plot(ds.time, rain, label='Rain Rate')
plt.xlabel('')
plt.ylabel('[mm/hr]')
plt.legend()
plt.subplot(6,1,5)
ds['evaporation_rate'].plot(label='Evaporation Rate')
plt.legend()
plt.ylabel('[mm/hr]')
plt.subplot(6,1,6)
plt.plot(ds.time,P,label='Air pressure')
plt.ylabel('[mbar]')
plt.legend()

plt.suptitle('SAFARI Mooring Met and evaporation')
# %%
