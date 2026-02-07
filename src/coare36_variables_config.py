import copy

reference_height = 3.0 #m, change the reference height here

variables_info = {
    # 1. friction velocity (usr)
    "friction_velocity": 
        {"units": "m/s", 
        "standard_name": "friction_velocity",
        "long_name": "friction velocity that includes gustiness",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 2.0
        },
    
    # 2. wind stress (tau)
    "wind_stress": {"units": "N/m^2", 
                    "standard_name": "magnitude_of_surface_downward_stress",
                    "long_name": "wind stress magnitude",
                    "coverage_content_type": "physicalMeasurement",
                    "valid_min": 0.0,
                    "valid_max": 5.0
                    },
    
    # 3. Sensible Heat Flux (hsb)
    "sensible_heat_flux": { 
        "units": "W/m^2", 
        "standard_name": "surface_downward_sensible_heat_flux",
        "long_name": "Sensible heat flux, positive when heating the ocean",
        "reverse_sign": True,
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -500.0,
        "valid_max": 500.0
    }, 
    
    # 4. Latent Heat Flux (hlb)
    "latent_heat_flux": {
        "units": "W/m^2", 
        "standard_name": "surface_downward_latent_heat_flux",
        "long_name": "Latent heat flux, positive when heating the ocean",
        "reverse_sign": True,
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -1000.0,
        "valid_max": 500.0
    },
    
    # 5. Atmospheric Buoyancy Flux (hbb)
    "atmospheric_buoyancy_flux": 
        {"units": "m^2/s^3", 
        "standard_name": "surface_buoyancy_flux_into_air",
        "long_name": "atmospheric buoyancy flux",
        "description": "positive when sensible and latent heat fluxes heat the ocean",
        "reverse_sign": True,
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -0.01,
        "valid_max": 0.01
        },
    
    # 6. Atmospheric Buoyancy Flux from Sonic (hsbb)
    "sonic_atmospheric_buoyancy_flux": {
        "units": "W/m^2", 
        "long_name": "atmospheric buoyancy flux from sonic",
        "description": "positive when sensible and latent heat fluxes heat the ocean",
        "reverse_sign": True,
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -500.0,
        "valid_max": 500.0
        },
    
    # 7. Latent Heat Flux webb correction (hlwebb)
    "latent_heat_flux_Webb_correction": {
        "units": "W/m^2",  
        "long_name": "Webb correction for latent heat flux",
        'description': "positive when heating the ocean",
        "reverse_sign": True,
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -100.0,
        "valid_max": 100.0
        },
    
    # 8. temperature scaling parameter (tsr)
    "temperature_scaling_parameter": 
        {"units": "K",
        "long_name": "temperature scaling parameter",
        "coverage_content_type": "auxiliaryInformation",
        "valid_min": -5.0,
        "valid_max": 5.0
        },
    
    # 9. specific humidity scaling parameter (qsr)
    "specific_humidity_scaling_parameter": 
        {"units": "g/kg", 
         'standard_name': 'specific_humidity',
        "long_name": "specific humidity scaling parameter",
        "coverage_content_type": "auxiliaryInformation",
        "valid_min": -10.0,
        "valid_max": 10.0
        },
    
    # 10. momentum roughness length
    "momentum_roughness_length": 
        {"units": "m", 
        "standard_name": "surface_roughness_length_for_momentum_in_air",
        "long_name": "momentum roughness length",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 0.1
        },
        
    # 11. thermal roughness length
    "thermal_roughness_length": 
        {"units": "m", 
        "standard_name": "surface_roughness_length_for_heat_in_air",
        "long_name": "thermal roughness length",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 0.1
        },
    
    # 12. moisture roughness length
    "moisture_roughness_length": 
        {"units": "m", 
        "standard_name": "surface_roughness_length_for_humidity_in_air",
        "long_name": "moisture roughness length",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 0.1
        },
    
    # 13. drag coefficient (Cd)
    "drag_coefficient": 
        {"units": "1", 
        "standard_name": "surface_drag_coefficient_for_momentum_in_air",
        "long_name": "wind stress transfer (drag) coefficient at height of the wind instrument",
        "coverage_content_type": "modelResult",
        "valid_min": 0.0,
        "valid_max": 0.01
        },
    
    # 14. stanton number (Ch)
    "stanton_number": 
        {"units": "1", 
        "standard_name": "surface_drag_coefficient_for_heat_in_air",
        "long_name": "sensible heat transfer coefficient (Stanton number)",
        "coverage_content_type": "modelResult",
        "valid_min": 0.0,
        "valid_max": 0.01
        },
    
    # 15. dalton number (Ce)
    "dalton_number": 
        {"units": "1", 
        "standard_name": "surface_drag_coefficient_for_humidity_in_air",
        "long_name": "latent heat transfer coefficient (Dalton number)",
        "coverage_content_type": "modelResult",
        "valid_min": 0.0,
        "valid_max": 0.01
        },
        
    # 16. Obukhov length
    "obukhov_length": 
        {"units": "m", 
        "standard_name": "atmosphere_boundary_layer_thickness",
        "long_name": "Monin-Obukhov length scale",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -1000.0,
        "valid_max": 1000.0
        },
        
    # 17. Monin-Obukhov stability parameter
    "stability_parameter": 
        {"units": "1", 
        "long_name": "Monin-Obukhov stability parameter zu/L",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -10.0,
        "valid_max": 10.0
        },
    
    # 18. cool-skin temperature depression
    "skin_temperature_dT": 
        {"units": "degC", 
         "standard_name": "difference_between_sea_surface_skin_temperature_and_sea_surface_subskin_temperature",
         "long_name": "cool-skin temperature depression",
         "description": "temperature difference between subskin and skin. negative value means skin is cooler than subskin",
        "reverse_sign": True,
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -2.0,
        "valid_max": 2.0
        },
    
    # 19. cool-skin humidity depression
    "skin_humidity_dq": {
        "units": "g/kg", 
        "long_name": "cool-skin humidity difference",
        "description": "humidity difference between subskin and skin. negative value means skin has lower humidity",
        "reverse_sign": True,
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -2.0,
        "valid_max": 2.0
        },

    # 20. cool-skin thickness
    "skin_thickness": {
        "units": "m", 
        "long_name": "cool-skin thickness",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 0.01
        },
    
    # 21. wind speed at reference height
    "wind_speed_at_reference_height": {
        "units": "m/s", 
        "standard_name": "wind_speed",
        "long_name": f"wind speed at {reference_height} m",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 50.0
        },
    
    # 22. air temperature at reference height
    "air_temperature_at_reference_height": {
        "units": "degC", 
        "standard_name": "air_temperature",
        "long_name": f"air temperature at {reference_height} meters",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -10.0,
        "valid_max": 40.0
        },
    
    # 23. specific humidity at reference height
    "specific_humidity_at_reference_height": {
        "units": "g/kg", 
        "standard_name": "specific_humidity",
        "long_name": f"air specific humidity at {reference_height} meters",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 30.0
        },
    
    # 24. relative humidity at reference height
    "relative_humidity_at_reference_height": {
        "units": "%", 
        "standard_name": "relative_humidity",
        "long_name": f"air relative humidity at {reference_height} meters",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 100.0
        },
    
    # 25. neutral wind speed at reference height
    "neutral_wind_speed_at_reference_height": {
        "units": "m/s", 
        "standard_name": "wind_speed",
        "long_name": f"neutral wind speed at {reference_height} meters",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 50.0
        },
    
    # 26. neutral air temperature at reference height
    "neutral_air_temperature_at_reference_height": {
        "units": "degC", 
        "standard_name": "air_temperature",
        "long_name": f"neutral value of air temperature at {reference_height} meters",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -10.0,
        "valid_max": 40.0
        },
    
    # 27. neutral specific humidity at reference height
    "neutral_specific_humidity_at_reference_height": {
        "units": "g/kg", 
        "standard_name": "specific_humidity",
        "long_name": f"neutral value of air specific humidity at {reference_height} meters",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 30.0
        },
    
    # 28. net longwave radiation
    "net_longwave_radiation": 
        {"units": "W/m^2", 
        "standard_name": "net_downward_longwave_flux_in_air",
        "long_name": "Net infrared radiation, positive heating ocean",
        "coverage_content_type": "radiation",
        "valid_min": -200.0,
        "valid_max": 200.0
        },
        
    # 29. net shortwave radiation
    "net_solar_radiation": 
        {"units": "W/m^2", 
        "standard_name": "surface_net_downward_shortwave_flux",
        "long_name": "Net solar radiation, positive heating ocean",
        "coverage_content_type": "radiation",
        "valid_min": 0.0,
        "valid_max": 1500.0
        },
    
    # 30. latent heat of vaporization
    "latent_heat_of_vaporization": 
        {"units": "J/kg", 
        "standard_name": "surface_latent_heat_flux_due_to_evaporation",
        "long_name": "latent heat of vaporization",
        "description": "positive when heating the ocean",
        "reverse_sign": True,
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 2.0e6,
        "valid_max": 2.5e6
        },
    
    # 31. air density at reference height
    "air_density_at_reference_height": {
        "units": "kg/m^3", 
        "standard_name": "air_density",
        "long_name": "density of air at reference height",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 1.0,
        "valid_max": 1.5
        },
    
    # 32. neutral wind speed
    "neutral_wind_speed": 
        {"units": "m/s", 
        "standard_name": "wind_speed",
        "long_name": "neutral value of wind speed at zu",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 50.0
        },
    
    # 33. wind speed adjusted to 10 m
    "wind_speed_10_meters": {
        "units": "m/s",
        "standard_name": "wind_speed",
        "long_name": "wind speed adjusted to 10 m",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 50.0
        },
    
    # 34. neutral wind speed adjusted to 10 m
    "neutral_wind_speed_10_meters": 
        {"units": "m/s", 
        "standard_name": "wind_speed",
        "long_name": "neutral value of wind speed at 10 m",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 50.0
        },
    
    # 35. drag coefficient at 10 m
    "neutral_drag_coefficient_10_meters": 
        {"units": "1", 
        "standard_name": "surface_drag_coefficient_in_air",
        "long_name": "neutral value of drag coefficient at 10m",
        "coverage_content_type": "modelResult",
        "valid_min": 0.0,
        "valid_max": 0.01
        },
        
    # 36. stanton number at 10 m
    "neutral_stanton_number_10_meters": 
        {"units": "1",
         "standard_name": "surface_drag_coefficient_for_heat_in_air",
         "long_name": "neutral value of Stanton number at 10m",
         "coverage_content_type": "modelResult",
         "valid_min": 0.0,
         "valid_max": 0.01
         },
    
    # 37. dalton number at 10 m
    "neutral_dalton_number_10_meters": 
        {"units": "1", 
        "standard_name": "surface_drag_coefficient_for_humidity_in_air",
        "long_name": "neutral value of Dalton number at 10m",
        "coverage_content_type": "modelResult",
        "valid_min": 0.0,
        "valid_max": 0.01
        },
    
    # 38. Rain heat flux
    "rain_heat_flux": 
        {"units": "W/m^2", 
         "standard_name": "heat_flux_into_sea_water_due_to_rainfall",
        "long_name": "rain heat flux",
        "description": "positive when rain heats the ocean",
        "reverse_sign": True,
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -100.0,
        "valid_max": 100.0
        },
    
    # 39. sea surface specific humidity
    "sea_surface_specific_humidity": {
        "units": "g/kg", 
        "standard_name": "surface_specific_humidity",
        "long_name": "sea surface specific humidity, assuming saturation",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 30.0
        },
        
    # 40. evaporation rate
    "evaporation_rate": 
        {"units": "mm/h", 
         "standard_name": "lwe_water_evaporation_rate",
        "long_name": "evaporation rate",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 10.0
        },
    
    # 41. air temperature at 10 meters
    "air_temperature_10_meters": 
        {"units": "degC", 
        "standard_name": "air_temperature",
        "long_name": "air temperature at 10m",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -10.0,
        "valid_max": 40.0
        },
    
    "neutral_air_temperature_10_meters": 
        {"units": "degC", 
        "standard_name": "air_temperature",
        "long_name": "neutral air temperature at 10m",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": -10.0,
        "valid_max": 40.0
        },
        
    "specific_humidity_10_meters": 
        {"units": "g/kg", 
        "standard_name": "specific_humidity",
        "long_name": "air specific humidity at 10m",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 30.0
        },
    
    "neutral_specific_humidity_10_meters": 
        {"units": "g/kg", 
        "standard_name": "specific_humidity",
        "long_name": "neutral air specific humidity at 10m",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 30.0
        },
        
    "relative_humidity_10_meters": 
        {"units": "%", 
        "standard_name": "relative_humidity",
        "long_name": "air relative humidity at 10m",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 100.0
        },
    
    "air_pressure_10_meters": 
        {"units": "mb", 
        "standard_name": "air_pressure",
        "long_name": "air pressure at 10m",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 900.0,
        "valid_max": 1100.0
        },
    
    "air_density_10_meters": 
        {"units": "kg/m^3", 
        "standard_name": "air_density",
        "long_name": "air density at 10m",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 1.0,
        "valid_max": 1.5
        },
    
    "gustiness_velocity": {
        "units": "m/s", 
        "standard_name": "wind_speed_of_gust",
        "long_name": "gustiness velocity",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 10.0
        },
    
    "whitecap_fraction": {
        "units": "1", 
        "long_name": "whitecap fraction",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 1.0
        },

    "dissipated_energy": {  # Changed name from dissipated_energy to energy_dissipation
        "units": "W/m^2", 
        "long_name": "energy dissipation by wave breaking",
        "coverage_content_type": "physicalMeasurement",
        "valid_min": 0.0,
        "valid_max": 1000.0
        },

    # Wind stress eastward component
    "wind_stress_eastward": {
        "units": "N/m^2", 
        "standard_name": "surface_downward_eastward_stress",
        "long_name": "eastward wind stress component",
        "coverage_content_type": "physicalMeasurement",
        "description": "Eastward component of wind stress. Positive values indicate stress acting towards the east",
        "valid_min": -5.0,
        "valid_max": 5.0
        },
    
    # Wind stress northward component
    "wind_stress_northward": {
        "units": "N/m^2", 
        "standard_name": "surface_downward_northward_stress",
        "long_name": "northward wind stress component",
        "coverage_content_type": "physicalMeasurement",
        "description": "Northward component of surface wind stress. Positive values indicate stress acting towards the north",
        "valid_min": -5.0,
        "valid_max": 5.0
        },

    # Optional outputs appended when return_albedo=True
    "albedo": {
        "units": "1",
        "standard_name": "surface_albedo",
        "long_name": "shortwave albedo (Payne 1972 or constant)",
        "coverage_content_type": "radiation",
        "valid_min": 0.0,
        "valid_max": 1.0
        },

    "shortwave_transmittance": {
        "units": "1",
        "long_name": "shortwave transmission factor",
        "description": "T_sw = min(2, sw_dn/solarmax_sw) from Payne (1972)",
        "coverage_content_type": "radiation",
        "valid_min": 0.0,
        "valid_max": 2.0
        },

    "solarmax_shortwave": {
        "units": "W/m^2",
        "long_name": "maximum shortwave radiation at top of atmosphere",
        "description": "solarmax_sw from Payne (1972)",
        "coverage_content_type": "radiation",
        "valid_min": 0.0,
        "valid_max": 2000.0
        },

    "solar_altitude_angle": {
        "units": "degrees",
        "long_name": "solar altitude angle",
        "description": "psi_sw from Payne (1972)",
        "coverage_content_type": "auxiliaryInformation",
        "valid_min": -90.0,
        "valid_max": 90.0
        },
    }


def _normalize_albedo_method(albedo_method):
    if albedo_method is None:
        return None
    if not isinstance(albedo_method, str):
        raise TypeError("albedo_method must be a string or None")
    key = albedo_method.strip().lower()
    if key in {"payne1972", "payne_1972", "payne-1972", "payne"}:
        return "Payne1972"
    if key in {"constant", "const"}:
        return "constant"
    raise ValueError("albedo_method must be 'Payne1972' or 'constant'")


def get_variables_info(albedo_method=None):
    """Return a deep copy of variables_info with optional albedo metadata tweaks."""
    info = copy.deepcopy(variables_info)
    method = _normalize_albedo_method(albedo_method)
    if method is None:
        return info
    albedo_meta = info.get("albedo")
    if albedo_meta is None:
        return info
    if method == "Payne1972":
        albedo_meta["long_name"] = "shortwave albedo (Payne 1972)"
        albedo_meta["description"] = (
            "Computed with zenith-angle varying albedo from Payne (1972)."
        )
    else:
        albedo_meta["long_name"] = "shortwave albedo (constant 0.055)"
        albedo_meta["description"] = (
            "Constant shortwave albedo of 0.055 (net factor 0.945)."
        )
    albedo_meta["comment"] = f"albedo_method={method}"
    return info
## check if the outputs match

## difference_between_sea_surface_skin_temperature_and_sea_surface_subskin_temperature
# This variable quantifies the temperature difference between the skin temperature (sea_surface_skin_temperature) 
# and the subskin temperature (sea_surface_subskin_temperature) 
# due to the turbulent and radiative heat fluxes at the air-sea interface. 
# This difference is commonly referred to as the “cool skin effect” 
# as the solar radiation absorbed within the very thin thermal subskin layer 
# is typically negligible compared to ocean surface heat loss 
# from the combined sensible, latent, and net longwave radiation heat fluxes.
