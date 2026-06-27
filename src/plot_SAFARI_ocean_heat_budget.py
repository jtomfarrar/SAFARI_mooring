# %% SAFARI_mooring - Ocean heat budget overview
# Created: 2026-06-25

# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import os
import sys
import gsw
from pathlib import Path

# %%
# Set working directory
home_dir = Path.home()
os.chdir(home_dir / 'Python/SAFARI_mooring/src')

# %%
ip = get_ipython() if "get_ipython" in globals() else None
if ip is not None:
    ip.run_line_magic("matplotlib", "widget")

plt.rcParams['figure.figsize'] = (9, 9)
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
ds_ww = xr.open_dataset(data_dir / 'SAFARI_wirewalker_profiles.nc')
ds_met = xr.open_dataset(data_dir / 'SAFARI_L3_met.nc')

t_start = max(ds_flux.time.values[0], ds_ww.time.values[0], ds_met.time.values[0])
t_end = min(ds_flux.time.values[-1], ds_ww.time.values[-1], ds_met.time.values[-1])

ds_flux = ds_flux.sel(time=slice(t_start, t_end))
ds_ww = ds_ww.sel(time=slice(t_start, t_end))
ds_met = ds_met.sel(time=slice(t_start, t_end))

Qh = ds_flux.sensible_heat_flux
Qe = ds_flux.latent_heat_flux
Qsw = ds_flux.net_solar_radiation
Qlw = ds_flux.net_longwave_radiation
net_heat_flux = Qh + Qe + Qsw + Qlw
net_heat_flux.attrs = {
    'long_name': 'net surface heat flux',
    'units': 'W/m^2',
    'description': 'sensible + latent + net shortwave + net longwave; positive heats the ocean',
}

# %% Add buoy surface T/S and estimate mixed layer depth
mld_density_threshold = 0.5
mean_lat = float(ds_met.latitude.mean(skipna=True))
mean_lon = float(ds_met.longitude.mean(skipna=True))

ds_ww_aug = ds_ww.copy(deep=True)
surface_temperature = ds_met.sea_surface_temperature.interp(time=ds_ww_aug.time)
surface_salinity = ds_met.salinity.interp(time=ds_ww_aug.time)
surface_depths = ds_ww_aug.depth.isel(depth=slice(0, 2))

ds_ww_aug['temperature'].loc[dict(depth=surface_depths)] = surface_temperature.broadcast_like(
    ds_ww_aug.temperature.sel(depth=surface_depths)
)
ds_ww_aug['salinity'].loc[dict(depth=surface_depths)] = surface_salinity.broadcast_like(
    ds_ww_aug.salinity.sel(depth=surface_depths)
)

temperature_attrs = ds_ww_aug.temperature.attrs.copy()
salinity_attrs = ds_ww_aug.salinity.attrs.copy()
ds_ww_aug['temperature'] = ds_ww_aug.temperature.interpolate_na(dim='depth')
ds_ww_aug['salinity'] = ds_ww_aug.salinity.interpolate_na(dim='depth')
ds_ww_aug.temperature.attrs = temperature_attrs
ds_ww_aug.salinity.attrs = salinity_attrs
ds_ww_aug.temperature.attrs['surface_note'] = (
    'First two depth rows are from SAFARI_L3_met.nc sea_surface_temperature; '
    'interior NaNs are filled by linear interpolation in depth with no vertical extrapolation.'
)
ds_ww_aug.salinity.attrs['surface_note'] = (
    'First two depth rows are from SAFARI_L3_met.nc salinity; '
    'interior NaNs are filled by linear interpolation in depth with no vertical extrapolation.'
)

pressure_mat = np.tile(ds_ww_aug.pressure.values[:, np.newaxis], (1, ds_ww_aug.sizes['time']))
SA = gsw.SA_from_SP(ds_ww_aug.salinity.values, pressure_mat, mean_lon, mean_lat)
CT = gsw.CT_from_t(SA, ds_ww_aug.temperature.values, pressure_mat)
sigma0 = gsw.sigma0(SA, CT)

mld = np.full(ds_ww_aug.sizes['time'], np.nan)
depth = ds_ww_aug.depth.values
surface_sigma0 = sigma0[0, :]
for i in range(ds_ww_aug.sizes['time']):
    if np.isfinite(surface_sigma0[i]):
        crossing = np.where(sigma0[:, i] >= surface_sigma0[i] + mld_density_threshold)[0]
        if len(crossing) > 0:
            mld[i] = depth[crossing[0]]

ds_ww_aug['mixed_layer_depth'] = xr.DataArray(
    mld,
    dims=('time',),
    coords={'time': ds_ww_aug.time},
    attrs={
        'long_name': 'mixed layer depth',
        'units': 'm',
        'density_threshold': mld_density_threshold,
        'description': 'First depth where sigma0 exceeds surface sigma0 by density_threshold kg m-3',
    },
)

print(f'Mean position for GSW: {mean_lat:.4f} degN, {mean_lon:.4f} degE')
print(f'MLD finite profiles: {np.isfinite(mld).sum()} of {len(mld)}')

print(f'Flux time range: {ds_flux.time.values[0]} to {ds_flux.time.values[-1]}')
print(f'Wirewalker time range: {ds_ww.time.values[0]} to {ds_ww.time.values[-1]}')
print(f'Met time range: {ds_met.time.values[0]} to {ds_met.time.values[-1]}')

# %% Fixed-depth temperature budget
# This follows the first deterministic temperature-budget steps from
# BoB_TS_balance_buoy_v1.m, but uses the SAFARI Wirewalker profiles interpolated
# to the hourly flux time grid.
h = 150
rho = 1025
cp = 3990
seconds_per_day = 86400
budget_smoothing_days = 14
budget_smoothing_hours = int(budget_smoothing_days * 24)


def smooth_budget_term(da):
    return da.rolling(time=budget_smoothing_hours, center=True).mean()


# Average duplicate Wirewalker timestamps before interpolation; xarray requires
# a unique time index for interp, and duplicate profiles are close enough for
# this first fixed-depth budget.
ds_ww_budget = ds_ww_aug.groupby('time').mean(skipna=True).interp(time=ds_flux.time)
ds_budget = xr.Dataset(coords={'time': ds_flux.time})

depth_budget = np.sort(np.append(ds_ww_budget.depth.values[ds_ww_budget.depth.values < h], h))
temperature_budget = ds_ww_budget.temperature.interp(depth=depth_budget)
Tbar = np.trapz(temperature_budget.values, temperature_budget.depth.values, axis=0) / h
ds_budget['Tbar'] = xr.DataArray(
    Tbar,
    dims=('time',),
    coords={'time': ds_budget.time},
    attrs={'long_name': f'0-{h:g} m layer-average temperature', 'units': 'degree_C'},
)

Tt = np.full(ds_budget.sizes['time'], np.nan)
time_seconds = (ds_budget.time.values - ds_budget.time.values[0]) / np.timedelta64(1, 's')
Tbar_values = ds_budget.Tbar.values
Tt[1:-1] = (Tbar_values[2:] - Tbar_values[:-2]) / (time_seconds[2:] - time_seconds[:-2])
ds_budget['Tt'] = xr.DataArray(
    Tt,
    dims=('time',),
    coords={'time': ds_budget.time},
    attrs={'long_name': f'tendency of 0-{h:g} m layer-average temperature', 'units': 'degree_C s-1'},
)

ds_budget['QN'] = net_heat_flux.interp(time=ds_budget.time)
ds_budget.QN.attrs = {
    'long_name': 'net surface heat flux',
    'units': 'W m-2',
    'description': 'sensible + latent + net shortwave + net longwave; positive heats the ocean',
}
ds_budget['Qs'] = ds_flux.net_solar_radiation.interp(time=ds_budget.time)
ds_budget.Qs.attrs = ds_flux.net_solar_radiation.attrs

ds_budget['Qterm'] = ds_budget.QN / rho / cp / h
ds_budget.Qterm.attrs = {
    'long_name': 'surface heat flux contribution to layer-average temperature tendency',
    'units': 'degree_C s-1',
}

# Use Paulson-Simpson 1977, eqn. 4, and Jerlov type IA; PWP defaults use
# R_PS = 0.62, lambda1 = 0.6 m, lambda2 = 20 m.
R_PS = 0.62
lambda1 = 0.6
lambda2 = 20
I_pen_ratio = R_PS * np.exp(-h / lambda1) + (1 - R_PS) * np.exp(-h / lambda2)
ds_budget['I_pen_ratio'] = xr.DataArray(
    I_pen_ratio,
    attrs={
        'long_name': f'fraction of net shortwave radiation penetrating below {h:g} m',
        'units': '1',
    },
)

# Qpen is negative because it removes the fraction of net shortwave that
# penetrates below h from the surface-layer heat input.
ds_budget['Qpen'] = -ds_budget.Qs * ds_budget.I_pen_ratio / rho / cp / h
ds_budget.Qpen.attrs = {
    'long_name': 'penetrating shortwave correction to layer-average temperature tendency',
    'units': 'degree_C s-1',
}
ds_budget['Qterm_plus_Qpen'] = ds_budget.Qterm + ds_budget.Qpen
ds_budget.Qterm_plus_Qpen.attrs = {
    'long_name': 'surface heat flux plus penetrating shortwave contribution',
    'units': 'degree_C s-1',
}

Kz = 5e-5
depth_below_h = ds_ww_budget.depth.where(ds_ww_budget.depth >= h, drop=True).isel(depth=0)
depth_above_h = ds_ww_budget.depth.where(ds_ww_budget.depth <= h, drop=True).isel(depth=-1)
T_below_h = ds_ww_budget.temperature.sel(depth=depth_below_h)
T_above_h = ds_ww_budget.temperature.sel(depth=depth_above_h)
Tz_h = (T_below_h - T_above_h) / (float(depth_below_h) - float(depth_above_h))
ds_budget['Tz_h'] = xr.DataArray(
    Tz_h.values,
    dims=('time',),
    coords={'time': ds_budget.time},
    attrs={'long_name': f'vertical temperature gradient at {h:g} m', 'units': 'degree_C m-1'},
)

# BoB_TS_balance_buoy_v1.m used QminusH = Tz_minusH2 * 5e-5 / H2 as a
# vertical-diffusion proxy.
ds_budget['QminusH'] = ds_budget.Tz_h * Kz / h
ds_budget.QminusH.attrs = {
    'long_name': 'vertical-diffusion proxy contribution to layer-average temperature tendency',
    'units': 'degree_C s-1',
    'Kz': Kz,
}
ds_budget['Qsum'] = ds_budget.Qterm_plus_Qpen + ds_budget.QminusH
ds_budget.Qsum.attrs = {'long_name': 'sum of surface flux, penetrating shortwave, and vertical diffusion terms',
                        'units': 'degree_C s-1'}

# Deferred term from the MATLAB workflow:
# entr = That_minusH2 * diff(H2) / 3600 / H2 was used as an entrainment-like term.
# These are intentionally deferred until the A/E comparison is checked.

# %% MLD-following temperature budget
ds_budget_mld = xr.Dataset(coords={'time': ds_flux.time})
ds_budget_mld['H2'] = ds_ww_budget.mixed_layer_depth
ds_budget_mld.H2.attrs = {
    'long_name': 'mixed layer depth interpolated to flux time',
    'units': 'm',
}

depth_grid = ds_ww_budget.depth.values
temperature_mld = ds_ww_budget.temperature.values
H2 = ds_budget_mld.H2.values

Tbar_mld = np.full(ds_budget_mld.sizes['time'], np.nan)
Tz_h_mld = np.full(ds_budget_mld.sizes['time'], np.nan)
That_minusH_mld = np.full(ds_budget_mld.sizes['time'], np.nan)
for i, H_i in enumerate(H2):
    profile = temperature_mld[:, i]
    valid = np.isfinite(profile)
    if (
        np.isfinite(H_i)
        and H_i > 0
        and valid.sum() >= 2
        and depth_grid[valid][0] <= 0
        and depth_grid[valid][-1] >= H_i
    ):
        valid_depth = depth_grid[valid]
        valid_temperature = profile[valid]
        valid_layer_depth = valid_depth[(valid_depth > 0) & (valid_depth < H_i)]
        layer_depth = np.sort(np.unique(np.append(valid_layer_depth, [0, H_i])))
        layer_temperature = np.interp(layer_depth, valid_depth, valid_temperature)
        Tbar_mld[i] = np.trapz(layer_temperature, layer_depth) / H_i

        Tz_profile = np.gradient(valid_temperature, valid_depth)
        Tz_h_mld[i] = np.interp(H_i, valid_depth, Tz_profile)
        That_minusH_mld[i] = np.interp(H_i, valid_depth, valid_temperature) - Tbar_mld[i]

ds_budget_mld['Tbar'] = xr.DataArray(
    Tbar_mld,
    dims=('time',),
    coords={'time': ds_budget_mld.time},
    attrs={'long_name': '0-MLD layer-average temperature', 'units': 'degree_C'},
)

Tt_mld = np.full(ds_budget_mld.sizes['time'], np.nan)
Tbar_mld_values = ds_budget_mld.Tbar.values
Tt_mld[1:-1] = (Tbar_mld_values[2:] - Tbar_mld_values[:-2]) / (time_seconds[2:] - time_seconds[:-2])
ds_budget_mld['Tt'] = xr.DataArray(
    Tt_mld,
    dims=('time',),
    coords={'time': ds_budget_mld.time},
    attrs={'long_name': 'tendency of MLD layer-average temperature', 'units': 'degree_C s-1'},
)

ds_budget_mld['QN'] = ds_budget.QN
ds_budget_mld['Qs'] = ds_budget.Qs
ds_budget_mld['Qterm'] = ds_budget_mld.QN / rho / cp / ds_budget_mld.H2
ds_budget_mld.Qterm.attrs = {
    'long_name': 'surface heat flux contribution to MLD layer-average temperature tendency',
    'units': 'degree_C s-1',
}
I_pen_1 = R_PS * np.exp(-ds_budget_mld.H2 / lambda1)
I_pen_2 = (1 - R_PS) * np.exp(-ds_budget_mld.H2 / lambda2)
ds_budget_mld['I_pen_ratio'] = I_pen_1 + I_pen_2
ds_budget_mld.I_pen_ratio.attrs = {
    'long_name': 'fraction of net shortwave radiation penetrating below mixed layer depth',
    'units': '1',
}
ds_budget_mld['Qpen'] = -ds_budget_mld.Qs * ds_budget_mld.I_pen_ratio / rho / cp / ds_budget_mld.H2
ds_budget_mld.Qpen.attrs = {
    'long_name': 'penetrating shortwave correction to MLD layer-average temperature tendency',
    'units': 'degree_C s-1',
}
ds_budget_mld['Qterm_plus_Qpen'] = ds_budget_mld.Qterm + ds_budget_mld.Qpen
ds_budget_mld.Qterm_plus_Qpen.attrs = {
    'long_name': 'surface heat flux plus penetrating shortwave contribution',
    'units': 'degree_C s-1',
}
ds_budget_mld['Tz_h'] = xr.DataArray(
    Tz_h_mld,
    dims=('time',),
    coords={'time': ds_budget_mld.time},
    attrs={'long_name': 'vertical temperature gradient at mixed layer depth', 'units': 'degree_C m-1'},
)
ds_budget_mld['QminusH'] = ds_budget_mld.Tz_h * Kz / ds_budget_mld.H2
ds_budget_mld.QminusH.attrs = {
    'long_name': 'vertical-diffusion proxy contribution to MLD layer-average temperature tendency',
    'units': 'degree_C s-1',
    'Kz': Kz,
}
ds_budget_mld['That_minusH'] = xr.DataArray(
    That_minusH_mld,
    dims=('time',),
    coords={'time': ds_budget_mld.time},
    attrs={'long_name': 'temperature anomaly at mixed layer depth relative to layer mean', 'units': 'degree_C'},
)

dh_dt = np.gradient(H2, time_seconds)
ds_budget_mld['dh_dt'] = xr.DataArray(
    dh_dt,
    dims=('time',),
    coords={'time': ds_budget_mld.time},
    attrs={'long_name': 'mixed layer depth tendency', 'units': 'm s-1'},
)
ds_budget_mld['entrainment_tendency'] = ds_budget_mld.That_minusH * ds_budget_mld.dh_dt / ds_budget_mld.H2
ds_budget_mld.entrainment_tendency.attrs = {
    'long_name': 'entrainment-like contribution to MLD layer-average temperature tendency',
    'units': 'degree_C s-1',
    'description': 'That_minusH * dh_dt / H2, following BoB_TS_balance_buoy_v1.m',
}

ds_budget_mld['Qsum'] = ds_budget_mld.Qterm_plus_Qpen + ds_budget_mld.QminusH + ds_budget_mld.entrainment_tendency
ds_budget_mld.Qsum.attrs = {
    'long_name': 'sum of surface flux, penetrating shortwave, vertical diffusion, and entrainment-like terms',
    'units': 'degree_C s-1',
}

# %% MLD-following temperature error budget
delHonH = 0.08
delTinstr = 0.01
delQ = 8
delta_t = 3600
del_FonF = 1
Nav = budget_smoothing_hours

H2_error = ds_budget_mld.H2.where(ds_budget_mld.H2 > 0)
Ntilde_TS = np.round(H2_error / 10)
Ntilde_TS = xr.where(Ntilde_TS < 1, 1, Ntilde_TS)
delH = delHonH * H2_error

er_Tbar = delTinstr / np.sqrt(Ntilde_TS) + np.abs(ds_budget_mld.That_minusH) * delHonH
ds_budget_mld['er_Tt'] = 2 * smooth_budget_term(er_Tbar) / (Nav * delta_t)
ds_budget_mld.er_Tt.attrs = {
    'long_name': 'uncertainty estimate for MLD layer-average temperature tendency',
    'units': 'degree_C s-1',
}

er_Qterm_raw = delQ / (rho * cp * H2_error) + np.abs(ds_budget_mld.Qterm) * delHonH
ds_budget_mld['er_Qterm'] = np.sqrt(smooth_budget_term(er_Qterm_raw ** 2))
ds_budget_mld.er_Qterm.attrs = {
    'long_name': 'uncertainty estimate for surface heat flux contribution',
    'units': 'degree_C s-1',
}

er_Tinstr_h = (np.sqrt(Ntilde_TS) - 1) / np.sqrt(Ntilde_TS) * delTinstr
er_H_h = delH * (ds_budget_mld.Tz_h - ds_budget_mld.That_minusH / H2_error)
er_That_h = er_Tinstr_h + er_H_h
er_dh_dt = 2 * delH / (Nav * delta_t)
er_entr_That = np.abs(er_That_h) * np.abs(ds_budget_mld.dh_dt / H2_error)
er_entr_dhdt = np.abs(ds_budget_mld.That_minusH / H2_error) * er_dh_dt
er_entr_H = delHonH * np.abs(ds_budget_mld.entrainment_tendency)
er_entrainment_raw = er_entr_That + er_entr_dhdt + er_entr_H
ds_budget_mld['er_entrainment_tendency'] = np.sqrt(smooth_budget_term(er_entrainment_raw ** 2 / Nav))
ds_budget_mld.er_entrainment_tendency.attrs = {
    'long_name': 'uncertainty estimate for entrainment-like contribution',
    'units': 'degree_C s-1',
}

er_QminusH_raw = del_FonF * np.abs(ds_budget_mld.QminusH)
ds_budget_mld['er_QminusH'] = np.sqrt(smooth_budget_term(er_QminusH_raw ** 2))
ds_budget_mld.er_QminusH.attrs = {
    'long_name': 'uncertainty estimate for vertical-diffusion proxy contribution',
    'units': 'degree_C s-1',
}

er_rhs_sum = ds_budget_mld.er_entrainment_tendency + ds_budget_mld.er_Qterm + ds_budget_mld.er_QminusH
ds_budget_mld['er_Qsum'] = er_rhs_sum / np.sqrt(2)
ds_budget_mld.er_Qsum.attrs = {
    'long_name': 'uncertainty estimate for MLD temperature budget RHS sum',
    'units': 'degree_C s-1',
}

# %% Plot heat fluxes, wind stress, and upper-ocean T/S
depth_max = 200
ds_ww_upper = ds_ww_aug.sel(depth=slice(0, depth_max))

fig = plt.figure(figsize=(9, 9))
gs = fig.add_gridspec(
    4, 2,
    height_ratios=[0.8, 0.7, 2, 2],
    width_ratios=[1, 0.03],
    hspace=0.12,
    wspace=0.04,
)
axs = np.array([fig.add_subplot(gs[i, 0]) for i in range(4)])
for ax in axs[1:]:
    ax.sharex(axs[0])
caxs = np.array([fig.add_subplot(gs[i, 1]) for i in range(4)])
for cax in caxs[:2]:
    cax.set_visible(False)

axs[0].plot(ds_flux.time, net_heat_flux, color='k', linewidth=1.4, label='Net')
axs[0].plot(ds_flux.time, ds_flux.sensible_heat_flux, color='C1', linewidth=1.0, label='Sensible')
axs[0].plot(ds_flux.time, ds_flux.latent_heat_flux, color='C0', linewidth=1.0, label='Latent')
axs[0].plot(ds_flux.time, ds_flux.net_solar_radiation, color='C3', linewidth=1.0, label='Net shortwave')
axs[0].plot(ds_flux.time, ds_flux.net_longwave_radiation, color='C4', linewidth=1.0, label='Net longwave')
axs[0].axhline(0, color='0.4', linewidth=0.8)
axs[0].set_ylabel('Heat flux\n(W m$^{-2}$)')
axs[0].legend(ncol=5, fontsize=7, loc='upper right')
axs[0].grid()

axs[1].plot(ds_flux.time, ds_flux.wind_stress, color='C2', linewidth=1.1)
axs[1].set_ylabel('Wind stress\n(N m$^{-2}$)')
axs[1].grid()

pcm0 = axs[2].pcolormesh(
    ds_ww_upper.time, ds_ww_upper.depth, ds_ww_upper.temperature, vmin=13.0, vmax=21,
    cmap='RdYlBu_r', shading='auto',
)
axs[2].set_ylim(depth_max, 0)
axs[2].set_ylabel('Depth (m)')
axs[2].plot(ds_ww_aug.time, ds_ww_aug.mixed_layer_depth, color='k', linewidth=1.2, label='MLD')
axs[2].legend(fontsize=7, loc='lower right')
plt.colorbar(pcm0, cax=caxs[2], label='Temperature ($^\\circ$C)')

pcm1 = axs[3].pcolormesh(
    ds_ww_upper.time, ds_ww_upper.depth, ds_ww_upper.salinity, vmin=34.2, vmax=34.7,
    cmap='viridis', shading='auto',
)
axs[3].set_ylim(depth_max, 0)
axs[3].set_ylabel('Depth (m)')
axs[3].plot(ds_ww_aug.time, ds_ww_aug.mixed_layer_depth, color='k', linewidth=1.2)
plt.colorbar(pcm1, cax=caxs[3], label='Salinity (psu)')

axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
axs[3].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
fig.autofmt_xdate()
fig.suptitle('SAFARI Ocean Heat Budget Overview', y=0.98)
axs[0].set_xlim(t_start, t_end)
if savefig:
    plt.savefig(__figdir__ / f'SAFARI_ocean_heat_budget_overview.{plotfiletype}', **savefig_args)


# %% Plot fixed-depth temperature budget
fig, ax = plt.subplots(figsize=(9, 4))

Tt_plot = smooth_budget_term(ds_budget.Tt * seconds_per_day)
Qterm_plot = smooth_budget_term(ds_budget.Qterm * seconds_per_day)
Qterm_plus_Qpen_plot = smooth_budget_term(ds_budget.Qterm_plus_Qpen * seconds_per_day)
QminusH_plot = smooth_budget_term(ds_budget.QminusH * seconds_per_day)
Qsum_plot = smooth_budget_term(ds_budget.Qsum * seconds_per_day)

ax.plot(ds_budget.time, Tt_plot, color='k', linewidth=1.4, label=r'dTbar/dt')
ax.plot(ds_budget.time, Qterm_plot, color='C0', linewidth=1.2, label='Surface flux')
ax.plot(ds_budget.time, Qterm_plus_Qpen_plot, color='C3', linewidth=1.2, label='Surface flux + penetrating solar')
ax.plot(ds_budget.time, QminusH_plot, color='C2', linewidth=1.1, label='vertical diffusion')
ax.plot(ds_budget.time, Qsum_plot, color='0.4', linewidth=1.4, linestyle='--', label='sum RHS')
ax.axhline(0, color='0.4', linewidth=0.8)
ax.set_ylabel('Temperature tendency\n($^\\circ$C day$^{-1}$)')
ax.grid()
ax.legend(fontsize=8, loc='best')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
fig.autofmt_xdate()
fig.suptitle(f'SAFARI Fixed-Depth Heat Budget, 0-{h:g} m', y=0.98)
plt.tight_layout()
if savefig:
    plt.savefig(__figdir__ / f'SAFARI_fixed_depth_heat_budget.{plotfiletype}', **savefig_args)


# %% Plot MLD-following temperature budget
fig, ax = plt.subplots(figsize=(9, 4))

Tt_mld_plot = smooth_budget_term(ds_budget_mld.Tt * seconds_per_day)
Qterm_mld_plot = smooth_budget_term(ds_budget_mld.Qterm * seconds_per_day)
Qterm_plus_Qpen_mld_plot = smooth_budget_term(ds_budget_mld.Qterm_plus_Qpen * seconds_per_day)
QminusH_mld_plot = smooth_budget_term(ds_budget_mld.QminusH * seconds_per_day)
entrainment_mld_plot = smooth_budget_term(ds_budget_mld.entrainment_tendency * seconds_per_day)
Qsum_mld_plot = smooth_budget_term(ds_budget_mld.Qsum * seconds_per_day)
time_mld_plot = ds_budget_mld.time.values

er_skip = 24 * 11
er_component_start = 0
er_sum_start = er_skip // 2
errorbar_args = {
    'capsize': 2,
    'elinewidth': 0.8,
}

ax.errorbar(time_mld_plot, Tt_mld_plot.values, yerr=(ds_budget_mld.er_Tt * seconds_per_day).values,
            fmt='-', color='k', linewidth=1.4, label=r'dTbar/dt',
            errorevery=(er_sum_start, er_skip), **errorbar_args)
ax.errorbar(time_mld_plot, Qterm_mld_plot.values, yerr=(ds_budget_mld.er_Qterm * seconds_per_day).values,
            fmt='-', color='C0', linewidth=1.2, label='Surface flux',
            errorevery=(er_component_start, er_skip), **errorbar_args)
ax.plot(time_mld_plot, Qterm_plus_Qpen_mld_plot.values, color='C3', linewidth=1.2,
        label='Surface flux + penetrating solar')
ax.errorbar(time_mld_plot, QminusH_mld_plot.values, yerr=(ds_budget_mld.er_QminusH * seconds_per_day).values,
            fmt='-', color='C2', linewidth=1.1, label='vertical diffusion',
            errorevery=(er_component_start, er_skip), **errorbar_args)
ax.errorbar(time_mld_plot, entrainment_mld_plot.values,
            yerr=(ds_budget_mld.er_entrainment_tendency * seconds_per_day).values,
            fmt='-', color='m', linewidth=1.1, label='entrainment',
            errorevery=(er_component_start, er_skip), **errorbar_args)
ax.errorbar(time_mld_plot, Qsum_mld_plot.values, yerr=(ds_budget_mld.er_Qsum * seconds_per_day).values,
            fmt='--', color='0.4', linewidth=1.4, label='sum RHS',
            errorevery=(er_sum_start, er_skip), **errorbar_args)
ax.axhline(0, color='0.4', linewidth=0.8)
ax.set_ylabel('Temperature tendency\n($^\\circ$C day$^{-1}$)')
ax.grid()
ax.legend(fontsize=8, loc='best')
ax.set_xlim(t_start, t_end)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
fig.autofmt_xdate()
fig.suptitle('SAFARI MLD-Following Heat Budget', y=0.98)
plt.tight_layout()
if savefig:
    plt.savefig(__figdir__ / f'SAFARI_mld_heat_budget.{plotfiletype}', **savefig_args)
