
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
        Fetches 'Deep' Environmental Data (10+ Parameters).
        Source: Open-Meteo Archive (Satellite/Reanalysis).
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": [
                "temperature_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall",
                "weather_code", "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
                "soil_temperature_0_7cm", "soil_moisture_0_7cm", "vapor_pressure_deficit"
            ],
            "timezone": "America/Chicago"
        }
        
        try:
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            hourly = response.Hourly()
            
            def get_col(idx): return hourly.Variables(idx).ValuesAsNumpy()
            
            data = {"date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )}
            
            # 12-Factor Environmental Vector
            data["temp"] = get_col(0)
            data["humidity"] = get_col(1)
            data["precip"] = get_col(2)
            data["rain"] = get_col(3)
            data["snow"] = get_col(4)      # TRAFFIC FACTOR
            data["clouds"] = get_col(6)    # VISIBILITY/SAT FACTOR
            data["wind_spd"] = get_col(7)
            data["wind_gust"] = get_col(8) # ROLLOVER/POLICE CLOSURE FACTOR
            data["soil_temp"] = get_col(9)
            data["soil_moist"] = get_col(10) # HARVEST FACTOR
            data["vpd"] = get_col(11)        # CROP STRESS FACTOR
            
            df = pd.DataFrame(data=data)
            
            # Aggregate to Daily (Max Risk / Mean Condition)
            df['date'] = df['date'].dt.date
            df_daily = df.groupby('date').agg({
                'temp': 'min', 'humidity': 'mean', 'precip': 'sum',
                'rain': 'sum', 'snow': 'sum', 'clouds': 'mean',
                'wind_spd': 'max', 'wind_gust': 'max',
                'soil_temp': 'min', 'soil_moist': 'mean', 'vpd': 'mean'
            }).reset_index()
            
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            return df_daily
            
        except Exception as e:
            print(f"Deep Weather API Error: {e}")
            return pd.DataFrame()

    def get_market_data(self, ticker="DC=F", start_date="2024-01-01", end_date="2024-03-31"):
        """
        Fetches Target Commodity + MACRO CONTEXT (Oil, Bonds, SPX).
        Implements rigorous Forward-Filling to handle weekends.
        """
        # List of tickers to fetch
        # CL=F: Crude Oil (Logistics Cost)
        # ^TNX: 10-Yr Treasury (Macro)
        # ^GSPC: S&P 500 (Sentiment)
        tickers = [ticker, "CL=F", "^TNX", "^GSPC"]
        
        try:
            # Download all at once
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker', timeout=10)
            
            if data.empty:
                return pd.DataFrame()
            
            # Helper to extract Close price for a ticker
            def extract_close(t):
                if isinstance(data.columns, pd.MultiIndex):
                    try:
                        return data[t]['Close']
                    except KeyError:
                        return pd.Series()
                else:
                    return data['Close'] if t == ticker else pd.Series()

            df_main = pd.DataFrame()
            
            # Primary Ticker
            s_main = extract_close(ticker)
            if s_main.empty:
                # Try simple naming if group_by failed
                if ticker in data.columns: s_main = data[ticker]
                elif 'Close' in data.columns: s_main = data['Close']
            
            if s_main.empty:
                print(f"CRITICAL: Could not find price for {ticker}")
                return pd.DataFrame()

            df_main['date'] = s_main.index
            df_main['price'] = s_main.values
            
            # Aux Tickers (Macro Context)
            s_oil = extract_close("CL=F")
            s_bond = extract_close("^TNX")
            s_spx = extract_close("^GSPC")
            
            # Alignment is automatic by index if we used join, but let's be safe with lists
            # We need to reindex everything to the main ticker's dates
            df_main.set_index('date', inplace=True)
            
            df_main['oil_price'] = s_oil
            df_main['bond_yield'] = s_bond
            df_main['spx_level'] = s_spx
            
            # KEY FIX: FILL MISSING DATA (Weekends/Holidays carry over previous close)
            df_main = df_main.ffill().bfill()
            
            df_main = df_main.reset_index()
            return df_main
            
        except Exception as e:
            print(f"Macro Data Error: {e}")
            return pd.DataFrame()

    def merge_data(self, df_weather, df_market):
        """
        Merges Deep Weather and Macro Market Data.
        Inner Join -> Drop Date -> Fill NAs.
        """
        if df_weather.empty or df_market.empty:
            return pd.DataFrame()
            
        df = pd.merge(df_market, df_weather, on='date', how='inner')
        return df.dropna()

if __name__ == "__main__":
    feed = MarketFeed()
    w = feed.get_weather_data()
    m = feed.get_market_data()
    merged = feed.merge_data(w, m)
    print(merged.head())
