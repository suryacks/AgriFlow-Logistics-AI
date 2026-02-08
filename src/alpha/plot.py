
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from .feed import MarketFeed
import io

class AlphaPlotter:
    def __init__(self):
        self.feed = MarketFeed()

    def generate_correlation_chart(self, start_date="2024-01-01", end_date="2024-03-31"):
        # 1. Fetch Data
        df_weather = self.feed.get_weather_data(start_date=start_date, end_date=end_date)
        df_market = self.feed.get_market_data(start_date=start_date, end_date=end_date)
        df = self.feed.merge_data(df_weather, df_market)
        
        if df.empty:
            return None
            
        # 2. Setup Plot (Dual Axis)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Style
        plt.style.use('dark_background')
        ax1.set_facecolor('#0f172a')
        fig.patch.set_facecolor('#0f172a')
        
        # Plot Price (Line)
        color = '#4ade80' # Green
        ax1.set_xlabel('Date', color='white')
        ax1.set_ylabel('Class III Milk Futures ($)', color=color)
        ax1.plot(df['date'], df['price'], color=color, linewidth=2, label='Price')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.1)
        
        # Plot Snowfall (Bar, Inverted) on Secondary Axis
        ax2 = ax1.twinx()
        color = '#38bdf8' # Blue
        ax2.set_ylabel('Snowfall (cm)', color=color)
        ax2.bar(df['date'], df['snowfall'], color=color, alpha=0.3, width=0.8, label='Snowfall')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Highlight "Severe Events" (Temp < -10C)
        freeze_days = df[df['temperature'] < -10]
        if not freeze_days.empty:
            ax1.scatter(freeze_days['date'], freeze_days['price'], color='red', zorder=5, label='Deep Freeze (<-10C)')

        # Title
        plt.title('Logistics-to-Alpha: Milk Futures vs. Omaha Blizzard Events', color='white', pad=20)
        
        # Date Formatting
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.gcf().autofmt_xdate()
        
        # Save to Buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0)
        plt.close()
        return buf

if __name__ == "__main__":
    plotter = AlphaPlotter()
    plotter.generate_correlation_chart()
