
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
        Fetches historical weather + Satellite-derived metrics.
        Location: Omaha, NE (Logistics Hub).
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "snowfall", "cloud_cover", "soil_temperature_0_7cm"],
            "timezone": "America/Chicago"
        }
        
        try:
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            # Process Hourly Data
            hourly = response.Hourly()
            
            # Helper to get numpy array
            def get_col(idx): return hourly.Variables(idx).ValuesAsNumpy()
            
            hourly_data = {"date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )}
            
            hourly_data["temperature"] = get_col(0)
            hourly_data["snowfall"] = get_col(1)
            hourly_data["cloud_cover"] = get_col(2) # Satellite Proxy
            hourly_data["soil_moisture"] = get_col(3) # Satellite Proxy
            
            df_weather = pd.DataFrame(data=hourly_data)
            
            # Aggregate to Daily
            df_weather['date'] = df_weather['date'].dt.date
            df_daily = df_weather.groupby('date').agg({
                'temperature': 'min',      # Freeze risk
                'snowfall': 'sum',         # Traffic risk
                'cloud_cover': 'mean',     # Visibility risk
                'soil_moisture': 'mean'    # Harvesting risk
            }).reset_index()
            
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            return df_daily
            
        except Exception as e:
            print(f"Weather API Error: {e}")
            return pd.DataFrame()

    def get_market_data(self, ticker="DC=F", start_date="2024-01-01", end_date="2024-03-31"):
        """
        Fetches Daily Futures Price with robust retries.
        """
        try:
            # yfinance download
            df_market = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=10)
            
            if df_market.empty:
                print(f"Warning: No data for {ticker}.")
                return pd.DataFrame(columns=['date', 'price'])
                
            df_market = df_market.reset_index()
            
            # Flatten MultiIndex columns if present
            if isinstance(df_market.columns, pd.MultiIndex):
                # We want the level 1 ('DC=F') if level 0 is 'Close'
                # Or just flatten generically
                df_market.columns = [
                    col[0] if isinstance(col, tuple) else col 
                    for col in df_market.columns
                ]

            # Rename standard columns
            cols_map = {'Date': 'date', 'Close': 'price', 'Adj Close': 'price'}
            df_market.rename(columns=cols_map, inplace=True)
            
            # If 'price' is still missing, try finding the ticker name
            if 'price' not in df_market.columns:
                if ticker in df_market.columns:
                    df_market.rename(columns={ticker: 'price'}, inplace=True)
                elif len(df_market.columns) > 1:
                     # Assumption: 1st col is Date, 2nd is Close/Price
                     df_market['price'] = df_market.iloc[:, 1]
            
            df_market = df_market[['date', 'price']].dropna()
            df_market['date'] = pd.to_datetime(df_market['date'])
            
            return df_market
            
        except Exception as e:
            print(f"Market Data Error for {ticker}: {e}")
            return pd.DataFrame(columns=['date', 'price'])

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
