
import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import yfinance as yf
import datetime

class MarketFeed:
    def __init__(self):
        # OpenMeteo Setup
        self.cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        self.retry_session = retry(self.cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo = openmeteo_requests.Client(session=self.retry_session)

    def get_weather_data(self, lat=41.25861, lon=-95.93779, start_date="2024-01-01", end_date="2024-03-31"):
        """
        Fetches historical weather (Temp + Snowfall) for a specific location.
        Default: Omaha, NE (I-80 Chokepoint).
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "snowfall"],
            "timezone": "America/Chicago"
        }
        
        responses = self.openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # Process Hourly Data
        hourly = response.Hourly()
        hourly_temp = hourly.Variables(0).ValuesAsNumpy()
        hourly_snow = hourly.Variables(1).ValuesAsNumpy()
        
        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}
        hourly_data["temperature"] = hourly_temp
        hourly_data["snowfall"] = hourly_snow
        
        df_weather = pd.DataFrame(data=hourly_data)
        
        # Aggregate to Daily (Min Temp, Max Snow)
        df_weather['date'] = df_weather['date'].dt.date
        df_daily = df_weather.groupby('date').agg({
            'temperature': 'min', # We care about freeze!
            'snowfall': 'sum'     # Total snow accumulation
        }).reset_index()
        
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        return df_daily

    def get_market_data(self, ticker="DC=F", start_date="2024-01-01", end_date="2024-03-31"):
        """
        Fetches Daily Futures Price (Class III Milk).
        """
        df_market = yf.download(ticker, start=start_date, end=end_date)
        if df_market.empty:
            # Fallback for testing on weekends/holidays if yfinance fails
            # Or use substitute like 'LE=F' (Live Cattle) if Milk is illiquid
            print(f"Warning: No data for {ticker}. Trying Live Cattle (LE=F)...")
            df_market = yf.download("LE=F", start=start_date, end=end_date)
            
        df_market = df_market.reset_index()
        # Ensure col names are standard (yfinance sometimes returns MultiIndex)
        if isinstance(df_market.columns, pd.MultiIndex):
            df_market.columns = df_market.columns.get_level_values(0)
            
        # Clean
        df_market = df_market[['Date', 'Close']]
        df_market.columns = ['date', 'price']
        df_market['date'] = pd.to_datetime(df_market['date'])
        
        return df_market

    def merge_data(self, df_weather, df_market):
        """
        Merges Weather and Market Data on Date.
        """
        df = pd.merge(df_market, df_weather, on='date', how='inner')
        return df

if __name__ == "__main__":
    feed = MarketFeed()
    w = feed.get_weather_data()
    m = feed.get_market_data()
    merged = feed.merge_data(w, m)
    print(merged.head())
