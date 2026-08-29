VARIABLES = {
    "temperature": {"label": "Temperature", "unit": "°C"},
    "dew_point": {"label": "Dew point", "unit": "°C"},
    "pressure": {"label": "Surface pressure", "unit": "hPa"},
    "humidity": {"label": "Relative humidity", "unit": "%"},
    "precipitation": {"label": "Precipitation", "unit": "mm"},
    "wind_speed": {"label": "Wind speed", "unit": "m/s"},
    "wind_direction": {"label": "Wind direction", "unit": "°"},
    "u_wind": {"label": "Eastward wind", "unit": "m/s"},
    "v_wind": {"label": "Northward wind", "unit": "m/s"},
}

LIVE_VARIABLES = {
    name: metadata
    for name, metadata in VARIABLES.items()
    if name
    in {
        "temperature",
        "dew_point",
        "pressure",
        "humidity",
        "precipitation",
        "wind_speed",
        "wind_direction",
    }
}
