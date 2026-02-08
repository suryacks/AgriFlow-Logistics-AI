
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import datetime

def test_feed():
    print("--- Testing Satellite Feed ---")
    
    # 1. Setup Client
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # 2. Define Params (The complex 12-factor request)
    params = {
        "latitude": 41.25861,
        "longitude": -95.93779,
        "start_date": "2024-01-01",
        "end_date": "2024-01-10",
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall",
            "weather_code", "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
            "soil_temperature_0_7cm", "soil_moisture_0_7cm", "vapor_pressure_deficit"
        ],
        "timezone": "America/Chicago"
    }
    
    print(f"Requesting: {url}")
    print(f"Params: {params}")
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        print(f"Success! Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
        
        hourly = response.Hourly()
        print(f"Hourly Data Points: {hourly.Variables(0).ValuesAsNumpy().shape}")
        
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feed()
