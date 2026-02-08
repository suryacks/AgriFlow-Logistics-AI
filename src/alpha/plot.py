
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from .feed import MarketFeed
from src.logistics.simulation_engine import SimulationEngine # Bridge
import io

class AlphaPlotter:
    def __init__(self):
        self.feed = MarketFeed()
        self.sim_engine = SimulationEngine()

    def generate_correlation_chart(self, start_date="2024-01-01", end_date="2024-03-31", ticker="DC=F"):
        # 1. Fetch Data
        df_weather = self.feed.get_weather_data(start_date=start_date, end_date=end_date)
        df_market = self.feed.get_market_data(ticker=ticker, start_date=start_date, end_date=end_date)
        df = self.feed.merge_data(df_weather, df_market)
        
        if df.empty:
            return None
            
        # 2. Run AgriFlow Analysis on Peak Events
        # Find the day with max snow/min temp
        worst_day = df.loc[df['snowfall'].idxmax()]
        
        # Normalize inputs for Simulation
        # Snow (cm): 0 -> 0 intensity, 20cm -> 1.0 intensity
        snow_intensity = min(worst_day['snowfall'] / 20.0, 1.0) 
        # Temp (min): 0C -> 0 intensity, -20C -> 1.0 intensity (Heat Factor in Sim represents 'Extremity')
        temp_intensity = min(abs(min(worst_day['temperature'], 0)) / 20.0, 1.0)
        
        # Run Simulation
        # We treat "Snow" as "Traffic Chaos" and "Freeze" as "Equipment Failure" (mapped to Heat Factor generic stress)
        stress_score, _, _ = self.sim_engine.obtain_stress_score(traffic_intensity=snow_intensity, heat_intensity=temp_intensity)
        
        # 3. Setup Plot
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        # Style
        plt.style.use('dark_background')
        ax1.set_facecolor('#0f172a')
        fig.patch.set_facecolor('#0f172a')
        
        # Plot Price
        color = '#4ade80'
        ax1.set_xlabel('Date', color='white')
        ax1.set_ylabel(f'{ticker} Price ($)', color=color)
        ax1.plot(df['date'], df['price'], color=color, linewidth=2, label='Price')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.1)
        
        # Plot Snowfall
        ax2 = ax1.twinx()
        color = '#38bdf8'
        ax2.set_ylabel('Snowfall (cm)', color=color)
        ax2.bar(df['date'], df['snowfall'], color=color, alpha=0.3, label='Snowfall')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Annotation: AgriFlow Signal
        # If Stress Score is high (>50), mark it
        if stress_score > 30:
            # Find correlation?
            # Place annotation at the Peak Snow Event
            peak_date = worst_day['date']
            peak_price = worst_day['price'] if not pd.isna(worst_day['price']) else df['price'].mean()
            
            label_text = (f"AgriFlow Alert: {stress_score:.0f}/100 Stress\n"
                          f"Disruption: {snow_intensity*100:.0f}% Chaos\n"
                          f"Signal: {'STRONG BUY' if stress_score > 60 else 'WATCH'}")
            
            ax1.annotate(label_text, 
                         xy=(peak_date, peak_price), 
                         xytext=(peak_date, peak_price * 1.05),
                         arrowprops=dict(facecolor='#f87171', shrink=0.05),
                         color='white',
                         bbox=dict(boxstyle="round,pad=0.3", fc="#ef4444", alpha=0.8))

        plt.title(f'AgriAlpha: {ticker} vs. Logistics Disruption (Omaha Hub)', color='white', pad=20)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.gcf().autofmt_xdate()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0)
        plt.close()
        return buf

if __name__ == "__main__":
    plotter = AlphaPlotter()
    plotter.generate_correlation_chart()
