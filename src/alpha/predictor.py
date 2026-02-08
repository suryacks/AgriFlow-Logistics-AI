
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from .feed import MarketFeed
from datetime import timedelta

class AgriAlphaPredictor:
    def __init__(self):
        self.feed = MarketFeed()
        # UPGRADE: Using Gradient Boosting for better sequential pattern recognition
        self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
        
    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_features(self, df):
        """
        Creates 'Lag Features' so we only use PAST data to predict FUTURE price.
        Uses 12+ Signal Sources + Technical Indicators.
        """
        if 'date' not in df.columns:
             print("DEBUG: DF Missing Date Column:", df.columns)
             return pd.DataFrame()

        df = df.copy()
        df = df.sort_values('date')
        
        # 1. Market Features (Momentum & Trend)
        df['prev_close'] = df['price'].shift(1)
        df['return_1d'] = df['price'].pct_change(1).shift(1)
        df['volatility_5d'] = df['price'].pct_change(1).rolling(5).std().shift(1)
        
        # New Technicals
        df['sma_10'] = df['price'].rolling(10).mean().shift(1)
        df['sma_30'] = df['price'].rolling(30).mean().shift(1)
        df['rsi_14'] = self.calculate_rsi(df['price'].shift(1), 14)
        
        # 2. Macro Features (Cost of Capital/Logistics)
        df['oil_cost'] = df['oil_price'].shift(1)
        df['macro_sentiment'] = df['spx_level'].pct_change(5).shift(1)
        
        # 3. Environmental Analysis (The Satellite/Weather Alpha)
        # Using Fallback-Safe Column Access
        if 'snow' in df.columns:
            df['traffic_stress'] = (df['snow'].rolling(3).sum() * 2 + df['rain']).shift(1)
        else:
            df['traffic_stress'] = 0
            
        if 'wind_gust' in df.columns and 'precip' in df.columns:
            df['accident_risk'] = (df['wind_gust'] * df['precip']).shift(1)
        else:
            df['accident_risk'] = 0
            
        # Harvest Conditions (Check existence for fallback)
        df['field_access'] = df['soil_moist'].shift(1) if 'soil_moist' in df.columns else 0
        df['crop_stress'] = df['vpd'].shift(1) if 'vpd' in df.columns else 0
        df['visibility_impairment'] = df['clouds'].shift(1) if 'clouds' in df.columns else 0
        
        # Drop NaN (caused by rolling windows)
        df = df.dropna()
        return df

    def predict_single_event(self, target_date_str, ticker="DC=F"):
        """
        Simulates a 'Time Machine' prediction. Handles Weekends by snapping to nearest past trading day.
        """
        target_date = pd.to_datetime(target_date_str)
        
        # UPGRADE: 1 Year Context for Grid Boosting
        start_date = (target_date - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')
        
        w = self.feed.get_weather_data(start_date=start_date, end_date=end_date)
        m = self.feed.get_market_data(ticker, start_date=start_date, end_date=end_date)
        
        if m.empty: return {"error": f"Failed to fetch market data for {ticker}."}
        
        df = self.feed.merge_data(w, m)
        if df.empty: return {"error": "Date mismatch. System requires at least 1 overlapping trading day."}

        df_ml = self.prepare_features(df)
        
        # FIND NEAREST VALID TRADING DAY if Target is missing (Weekend/Holiday)
        # We look for the exact date, or the closest date *before* it if missing? 
        # Actually, for "prediction", we want the *next* trading day if selected is a weekend.
        # But for "hindsight analysis" (Time Machine), we strictly look for the record.
        
        record = df_ml[df_ml['date'] == target_date]
        if record.empty:
            # Fallback: Try previous Friday or next Monday? 
            # Let's just find the closest date available in the dataset
            # abs(df_ml['date'] - target_date).idxmin()
            pass
            return {"error": "Market Closed on this date. Please select a valid Trading Day (Mon-Fri)."}
        
        # Split Data
        train = df_ml[df_ml['date'] < target_date]
        if len(train) < 60: return {"error": "Insufficient historical data (Need >60 days)."}
        
        features = [
            'prev_close', 'return_1d', 'volatility_5d', 'sma_10', 'sma_30', 'rsi_14',
            'oil_cost', 'macro_sentiment', 'traffic_stress', 'accident_risk', 
            'field_access', 'crop_stress'
        ]
        
        # Filter available features (in case fallback dropped some)
        features = [f for f in features if f in df_ml.columns]
        
        self.model.fit(train[features], train['price'])
        
        predicted_price = self.model.predict(record[features])[0]
        actual_price = record['price'].values[0]
        prev_price = record['prev_close'].values[0]
        
        predicted_move = "UP" if predicted_price > prev_price else "DOWN"
        actual_move = "UP" if actual_price > prev_price else "DOWN"
        
        return {
            "date": target_date_str,
            "predicted_price": round(predicted_price, 2),
            "actual_price": round(actual_price, 2),
            "delta_percent": round(((predicted_price - actual_price)/actual_price)*100, 2),
            "signal": predicted_move,
            "actual_move": actual_move,
            "correct_direction": (predicted_move == actual_move),
            "satellite_data": {
                "traffic_stress_index": round(record.get('traffic_stress', 0).values[0], 1),
                "accident_risk_score": round(record.get('accident_risk', 0).values[0], 1),
                "soil_moisture": round(record.get('field_access', 0).values[0], 2),
                "visibility_pct": round(record.get('visibility_impairment', 0).values[0], 0)
            },
            "macro_data": {
                "oil_price": round(record.get('oil_cost', 0).values[0], 2)
            }
        }

    def run_backtest(self, start_date, end_date, ticker="DC=F"):
        """
        Walk-Forward Validation. Returns Equity Curve AND Asset Price History.
        """
        # Buffer 365 Days for Rolling Windows + Training
        buffer_start = (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
        
        w = self.feed.get_weather_data(start_date=buffer_start, end_date=end_date)
        m = self.feed.get_market_data(ticker, start_date=buffer_start, end_date=end_date)
        
        if m.empty: return {"error": f"Market Data Failed for {ticker}"}
        
        df = self.feed.merge_data(w, m)
        if df.empty: return {"error": "No overlapping data."}
        
        df_ml = self.prepare_features(df)
        
        eval_dates = df_ml[(df_ml['date'] >= pd.to_datetime(start_date)) & (df_ml['date'] <= pd.to_datetime(end_date))]
        
        correct_count = 0
        total_count = 0
        capital = 10000.0
        portfolio_history = []
        
        features = [
            'prev_close', 'return_1d', 'volatility_5d', 'sma_10', 'sma_30', 'rsi_14',
            'oil_cost', 'macro_sentiment', 'traffic_stress', 'accident_risk', 
            'field_access', 'crop_stress'
        ]
        # Safety filter
        features = [f for f in features if f in df_ml.columns]
        
        for idx, row in eval_dates.iterrows():
            curr_date = row['date']
            train = df_ml[df_ml['date'] < curr_date]
            
            # Need strict minimum for RSI/SMA
            if len(train) < 60: continue 
            
            # Re-train every day (Walk Forward)
            try:
                self.model.fit(train[features], train['price'])
                pred = self.model.predict(pd.DataFrame([row[features]]))[0]
                actual = row['price']
                param_prev = row['prev_close']
                
                # Logic: If Pred > Prev, Buy. If Pred < Prev, Sell.
                signal = 1 if pred > param_prev else -1
                
                # Calculate Daily ROI
                pct_change = (actual - param_prev) / param_prev
                
                # Sizing: 20% of Capital
                bet_size = capital * 0.20
                pnl = bet_size * signal * pct_change
                capital += pnl
                
                # Accuracy tracking
                pred_dir = 1 if pred > param_prev else -1
                act_dir = 1 if actual > param_prev else -1
                if pred_dir == act_dir: correct_count += 1
                total_count += 1
                
                portfolio_history.append({
                    'date': curr_date.strftime('%Y-%m-%d'), 
                    'equity': round(capital, 2),
                    'asset_price': round(actual, 2) # Added for Comparison
                })
            except Exception as e:
                print(f"Skipping date {curr_date}: {e}")
                continue
            
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        return {
            "accuracy": round(accuracy, 1),
            "final_capital": round(capital, 2),
            "roi": round(((capital - 10000)/10000)*100, 2),
            "total_trades": total_count,
            "curve": portfolio_history
        }
