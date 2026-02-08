
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .feed import MarketFeed
from datetime import timedelta

class AgriAlphaPredictor:
    def __init__(self):
        self.feed = MarketFeed()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def prepare_features(self, df):
        """
        Creates 'Lag Features' so we only use PAST data to predict FUTURE price.
        Uses 12+ Signal Sources.
        """
        df = df.copy()
        df = df.sort_values('date')
        
        # 1. Market Features (Momentum)
        df['prev_close'] = df['price'].shift(1)
        df['return_1d'] = df['price'].pct_change(1).shift(1)
        df['volatility_5d'] = df['price'].pct_change(1).rolling(5).std().shift(1)
        
        # 2. Macro Features (Cost of Capital/Logistics)
        df['oil_cost'] = df['oil_price'].shift(1)       # Transport Fuel
        df['macro_sentiment'] = df['spx_level'].pct_change(5).shift(1) # Market Mood
        
        # 3. Environmental Analysis (The Satellite/Weather Alpha)
        
        # A. TRAFFIC CHAOS INDEX (Snow + Rain + Rush Hour Stress Proxy)
        # We assume storms have lingering effects (3-day rolling sum of snow)
        df['traffic_stress'] = (df['snow'].rolling(3).sum() * 2 + df['rain']) .shift(1)
        
        # B. POLICE/ACCIDENT RISK (Wind Gusts + Precip)
        # High winds flip trucks. High precip causes crashes.
        # This proxy simulates "Force Majeure" events on the highway.
        df['accident_risk'] = (df['wind_gust'] * df['precip']).shift(1)
        
        # C. HARVEST CONDITIONS (Satellite Soil Moisture + VPD)
        # Can farmers even get into the field?
        df['field_access'] = df['soil_moist'].shift(1)
        df['crop_stress'] = df['vpd'].shift(1)
        
        # D. VISIBILITY (Satellite Cloud Cover)
        df['visibility_impairment'] = df['clouds'].shift(1)
        
        df = df.dropna()
        return df

    def predict_single_event(self, target_date_str, ticker="DC=F"):
        """
        Simulates a 'Time Machine' prediction using 12+ Sources.
        """
        # Buffer increased to 120 days for better training context
        target_date = pd.to_datetime(target_date_str)
        start_date = (target_date - timedelta(days=120)).strftime('%Y-%m-%d')
        end_date = (target_date + timedelta(days=5)).strftime('%Y-%m-%d')
        
        # Fetch Enhanced Data
        w = self.feed.get_weather_data(start_date=start_date, end_date=end_date)
        m = self.feed.get_market_data(ticker, start_date=start_date, end_date=end_date)
        
        if m.empty: 
            return {"error": f"Failed to fetch market data for {ticker}. Try a different date or ticker."}
        if w.empty:
            return {"error": "Failed to fetch satellite/weather data."}
        
        df = self.feed.merge_data(w, m)
        if df.empty:
            return {"error": "Date mismatch between Weather and Market data (likely Holiday/Weekend). Try a weekday."}

        df_ml = self.prepare_features(df)
        
        # Split
        train = df_ml[df_ml['date'] < target_date]
        test = df_ml[df_ml['date'] == target_date]
        
        if train.empty or test.empty:
            return {"error": "Insufficient historical data for this specific date range."}
            
        # Expanded Feature Set (The "Omniscient" Model)
        features = [
            'prev_close', 'return_1d', 'volatility_5d', 'oil_cost', 'macro_sentiment',
            'traffic_stress', 'accident_risk', 'field_access', 'crop_stress', 'visibility_impairment'
        ]
        
        X_train = train[features]
        y_train = train['price']
        X_test = test[features]
        
        self.model.fit(X_train, y_train)
        
        predicted_price = self.model.predict(X_test)[0]
        actual_price = test['price'].values[0]
        
        prev_price = test['prev_close'].values[0]
        predicted_move = "UP" if predicted_price > prev_price else "DOWN"
        actual_move = "UP" if actual_price > prev_price else "DOWN"
        correct = (predicted_move == actual_move)
        
        # Contextual Data for UI
        return {
            "date": target_date_str,
            "predicted_price": round(predicted_price, 2),
            "actual_price": round(actual_price, 2),
            "delta_percent": round(((predicted_price - actual_price)/actual_price)*100, 2),
            "signal": predicted_move,
            "actual_move": actual_move,
            "correct_direction": bool(correct),
            "satellite_data": {
                "traffic_stress_index": round(test['traffic_stress'].values[0], 1),
                "accident_risk_score": round(test['accident_risk'].values[0], 1),
                "soil_moisture": round(test['field_access'].values[0], 2),
                "visibility_pct": round(test['visibility_impairment'].values[0], 0)
            },
            "macro_data": {
                "oil_price": round(test['oil_cost'].values[0], 2)
            }
        }

    def run_backtest(self, start_date, end_date, ticker="DC=F"):
        """
        Walk-Forward Validation with Enhanced Features.
        """
        buffer_start = (pd.to_datetime(start_date) - timedelta(days=90)).strftime('%Y-%m-%d')
        
        w = self.feed.get_weather_data(start_date=buffer_start, end_date=end_date)
        m = self.feed.get_market_data(ticker, start_date=buffer_start, end_date=end_date)
        
        if m.empty: return {"error": "Market Data Failed"}
        
        df = self.feed.merge_data(w, m)
        df_ml = self.prepare_features(df)
        
        eval_dates = df_ml[(df_ml['date'] >= pd.to_datetime(start_date)) & (df_ml['date'] <= pd.to_datetime(end_date))]
        
        correct_count = 0
        total_count = 0
        capital = 10000.0
        portfolio_history = []
        
        features = [
            'prev_close', 'return_1d', 'volatility_5d', 'oil_cost', 'macro_sentiment',
            'traffic_stress', 'accident_risk', 'field_access', 'crop_stress', 'visibility_impairment'
        ]
        
        for idx, row in eval_dates.iterrows():
            curr_date = row['date']
            train = df_ml[df_ml['date'] < curr_date]
            if len(train) < 50: continue 
            
            self.model.fit(train[features], train['price'])
            pred = self.model.predict(pd.DataFrame([row[features]]))[0]
            actual = row['price']
            param_prev = row['prev_close']
            
            # Simple Trading Strategy
            signal = 0 
            if pred > param_prev * 1.002: signal = 1 # Lower threshold for more activity
            elif pred < param_prev * 0.998: signal = -1 
            
            pct_change = (actual - param_prev) / param_prev
            if signal != 0:
                pnl = capital * 0.20 * signal * pct_change # Aggressive 20% bets
                capital += pnl
            
            pred_dir = 1 if pred > param_prev else -1
            act_dir = 1 if actual > param_prev else -1
            if pred_dir == act_dir: correct_count += 1
            total_count += 1
            
            portfolio_history.append({'date': curr_date.strftime('%Y-%m-%d'), 'equity': round(capital, 2)})
            
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        return {
            "accuracy": round(accuracy, 1),
            "final_capital": round(capital, 2),
            "roi": round(((capital - 10000)/10000)*100, 2),
            "total_trades": total_count,
            "curve": portfolio_history
        }
