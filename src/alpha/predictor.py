
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
        """
        df = df.copy()
        df = df.sort_values('date')
        
        # 1. Market Features (Momentum)
        df['prev_close'] = df['price'].shift(1)
        df['return_1d'] = df['price'].pct_change(1).shift(1) # Return yesterday
        df['return_5d'] = df['price'].pct_change(5).shift(1) # Return last week
        
        # 2. Logistics/Weather Features (The Alpha)
        # We want to know if "Yesterday's Snow" impacts "Today's Price"
        df['snow_1d'] = df['snowfall'].shift(1)
        df['snow_3d_sum'] = df['snowfall'].rolling(3).sum().shift(1)
        df['temp_min_1d'] = df['temperature'].shift(1)
        df['soil_moisture_1d'] = df['soil_moisture'].shift(1)
        
        # 3. Target
        # We want to predict 'price' (Today) using the shifted columns above.
        
        df = df.dropna()
        return df

    def predict_single_event(self, target_date_str, ticker="DC=F"):
        """
        Simulates a 'Time Machine' prediction.
        1. Fetches data up to Target Date.
        2. Trains model on data < Target Date.
        3. Predicts Target Date.
        """
        # Fetch data buffer (3 months before to ensure enough training samples)
        target_date = pd.to_datetime(target_date_str)
        start_date = (target_date - timedelta(days=90)).strftime('%Y-%m-%d')
        end_date = (target_date + timedelta(days=5)).strftime('%Y-%m-%d') # Fetch a bit after to get GT
        
        # Fetch
        w = self.feed.get_weather_data(start_date=start_date, end_date=end_date)
        m = self.feed.get_market_data(ticker, start_date=start_date, end_date=end_date)
        if m.empty or w.empty: return {"error": "Insufficient Data"}
        
        df = self.feed.merge_data(w, m)
        df_ml = self.prepare_features(df)
        
        # Split: Train on EVERYTHING before target_date
        train = df_ml[df_ml['date'] < target_date]
        test = df_ml[df_ml['date'] == target_date]
        
        if train.empty or test.empty:
            return {
                "error": "Not enough historical data to predict this date.",
                "context": f"Train size: {len(train)}, Test Date in Data: {not test.empty}"
            }
            
        # Features
        features = ['prev_close', 'return_1d', 'snow_3d_sum', 'temp_min_1d', 'soil_moisture_1d']
        X_train = train[features]
        y_train = train['price']
        X_test = test[features]
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Predict
        predicted_price = self.model.predict(X_test)[0]
        actual_price = test['price'].values[0]
        
        # Logic
        prev_price = test['prev_close'].values[0]
        predicted_move = "UP" if predicted_price > prev_price else "DOWN"
        actual_move = "UP" if actual_price > prev_price else "DOWN"
        correct = (predicted_move == actual_move)
        
        # Sat Data Context
        snow_context = test['snow_3d_sum'].values[0]
        
        return {
            "date": target_date_str,
            "predicted_price": round(predicted_price, 2),
            "actual_price": round(actual_price, 2),
            "delta_percent": round(((predicted_price - actual_price)/actual_price)*100, 2),
            "signal": predicted_move,
            "actual_move": actual_move,
            "correct_direction": bool(correct),
            "satellite_data": {
                "snow_accumulation_cm": round(snow_context, 1),
                "soil_temp": round(test['soil_moisture_1d'].values[0], 1)
            }
        }

    def run_backtest(self, start_date, end_date, ticker="DC=F"):
        """
        Walk-Forward Validation.
        Re-trains the model every day to simulate real trading.
        """
        # Fetch ALL data once
        # Need buffer for lag features
        buffer_start = (pd.to_datetime(start_date) - timedelta(days=60)).strftime('%Y-%m-%d')
        
        w = self.feed.get_weather_data(start_date=buffer_start, end_date=end_date)
        m = self.feed.get_market_data(ticker, start_date=buffer_start, end_date=end_date)
        
        if m.empty: return {"error": "Market Data Failed"}
        
        df = self.feed.merge_data(w, m)
        df_ml = self.prepare_features(df)
        
        results = []
        
        # Iterate through target range
        eval_dates = df_ml[(df_ml['date'] >= pd.to_datetime(start_date)) & (df_ml['date'] <= pd.to_datetime(end_date))]
        
        correct_count = 0
        total_count = 0
        capital = 10000.0 # Virtual Portfolio
        portfolio_history = []
        
        features = ['prev_close', 'return_1d', 'snow_3d_sum', 'temp_min_1d', 'soil_moisture_1d']
        
        for idx, row in eval_dates.iterrows():
            curr_date = row['date']
            
            # Train on PAST only
            train = df_ml[df_ml['date'] < curr_date]
            if len(train) < 30: continue # Need minimum history
            
            self.model.fit(train[features], train['price'])
            
            # Predict
            pred = self.model.predict(pd.DataFrame([row[features]]))[0]
            actual = row['price']
            param_prev = row['prev_close']
            
            # Trade Logic
            signal = 0 # 1 Buy, -1 Sell
            if pred > param_prev * 1.005: signal = 1 # Predict >0.5% gain
            elif pred < param_prev * 0.995: signal = -1 # Predict >0.5% droo
            
            # PnL
            pct_change = (actual - param_prev) / param_prev
            pnl = 0
            if signal != 0:
                pnl = capital * 0.10 * signal * pct_change # Bet 10% of portfolio
                capital += pnl
            
            # Direction Accuracy
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
