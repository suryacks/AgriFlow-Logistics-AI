
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
        Try -> Fallback -> Fail Gracefully.
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Deep Param Set (12 Factors)
        full_params = {
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
        
        # Fallback Param Set (Critical Factors Only)
        basic_params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "rain", "snowfall", "wind_gusts_10m"],
            "timezone": "America/Chicago"
        }
        
        def process_response(response, is_basic=False):
            hourly = response.Hourly()
            
            data = {"date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )}
            
            def get_col(idx): 
                vals = hourly.Variables(idx).ValuesAsNumpy()
                if len(vals) != len(data["date"]):
                     # Pad or trunc if length mismatch (rare API quirk)
                     return pd.Series(vals).reindex(range(len(data["date"])), fill_value=0).values
                return vals
            
            if not is_basic:
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
            else:
                # Basic Fallback (Fill missing with defaults)
                data["temp"] = get_col(0)
                data["rain"] = get_col(1)
                data["snow"] = get_col(2)
                data["wind_gust"] = get_col(3)
                # Fill Missing
                zeros = [0.0] * len(data["date"])
                data["humidity"] = zeros; data["precip"] = data["rain"] # Approx
                data["clouds"] = zeros; data["wind_spd"] = zeros
                data["soil_temp"] = zeros; data["soil_moist"] = zeros; data["vpd"] = zeros

            df = pd.DataFrame(data=data)
            
            # Aggregate to Daily
            df['date'] = df['date'].dt.date
            df_daily = df.groupby('date').agg({
                'temp': 'min', 'humidity': 'mean', 'precip': 'sum',
                'rain': 'sum', 'snow': 'sum', 'clouds': 'mean',
                'wind_spd': 'max', 'wind_gust': 'max',
                'soil_temp': 'min', 'soil_moist': 'mean', 'vpd': 'mean'
            }).reset_index()
            
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            return df_daily

        try:
            # Attempt 1: Full Fidelity
            responses = self.openmeteo.weather_api(url, params=full_params)
            return process_response(responses[0], is_basic=False)
            
        except Exception as e:
            print(f"Deep Weather API Failed ({e}). Attempting Fallback...")
            try:
                # Attempt 2: minimal vital signs
                responses = self.openmeteo.weather_api(url, params=basic_params)
                print("Fallback Weather Data Retrieved.")
                return process_response(responses[0], is_basic=True)
            except Exception as e2:
                print(f"CRITICAL: All Weather feeds failed. {e2}")
                return pd.DataFrame()

    def get_market_data(self, ticker="DC=F", start_date="2024-01-01", end_date="2024-03-31"):
        """
        Fetches Target Commodity + MACRO CONTEXT.
        """
        try:
            # 1. Fetch Target Asset
            df_target = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df_target.empty:
                print(f"Warning: No data for {ticker}")
                return pd.DataFrame()

            # Flatten YF MultiIndex if present
            if isinstance(df_target.columns, pd.MultiIndex):
                # Dropping the 'Ticker' level, keeping 'Price Type'
                df_target.columns = df_target.columns.get_level_values(0)

            # Standardize
            df_target = df_target.reset_index()
            # YFinance sometimes returns 'Date' or 'Datetime'
            if 'Date' in df_target.columns:
                df_target.rename(columns={'Date': 'date'}, inplace=True)
            elif 'Datetime' in df_target.columns: # For intraday
                df_target.rename(columns={'Datetime': 'date'}, inplace=True)
                
            cols_map = {'Close': 'price', 'Adj Close': 'price'}
            df_target.rename(columns=cols_map, inplace=True)
            
            if 'price' not in df_target.columns:
                # If ticker name is the column
                if ticker in df_target.columns:
                     df_target.rename(columns={ticker: 'price'}, inplace=True)
                else:
                     # Fallback: Take last column
                     df_target['price'] = df_target.iloc[:, -1]

            df_target = df_target[['date', 'price']].copy()
            df_target['date'] = pd.to_datetime(df_target['date']).dt.tz_localize(None) # Remove TZ for easier merge

            # 2. Fetch Macro Context (Oil, Bond, SPX)
            # We fetch these separately to avoid the complex MultiIndex merging logic of bulk download
            # We assume these are highly liquid and always available
            macro_tickers = {"CL=F": "oil_price", "^TNX": "bond_yield", "^GSPC": "spx_level"}
            
            df_macro = yf.download(list(macro_tickers.keys()), start=start_date, end=end_date, progress=False)
            
            # Handle Macro MultiIndex
            # Structure: (Price, Ticker)
            if isinstance(df_macro.columns, pd.MultiIndex):
                # We want 'Close' for each ticker
                try:
                    df_macro = df_macro['Close']
                except:
                     pass # Fallback

            df_macro = df_macro.reset_index()
            if 'Date' in df_macro.columns: df_macro.rename(columns={'Date': 'date'}, inplace=True)
            
            df_macro['date'] = pd.to_datetime(df_macro['date']).dt.tz_localize(None)

            # Rename columns to our internal names
            df_macro.rename(columns=macro_tickers, inplace=True)
            
            # 3. Merge Target + Macro
            # Left join on Target to preserve our asset's timeline
            df_final = pd.merge(df_target, df_macro, on='date', how='left')
            
            # Fill Missing Macro Data (Weekends/Holidays)
            df_final = df_final.ffill().bfill()
            
            return df_final

        except Exception as e:
            print(f"Market Data Error: {e}")
            import traceback
            traceback.print_exc()
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
