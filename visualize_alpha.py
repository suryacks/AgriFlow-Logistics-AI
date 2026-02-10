
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import sys
import os

# Path Setup
sys.path.append(os.getcwd())

from src.alpha.bio_pricing import BiologicalDecay
from src.ingest.traffic_vision import TrafficVision

def generate_dynamic_alpha_thesis():
    """
    Generates a randomized, robust Alpha Thesis chart.
    Demonstrates:
    1. Invisible Friction (Bio-Shrink)
    2. Computer Vision Signals (Traffic)
    3. Portfolio Growth ($5k -> $Profit)
    """
    
    # 1. Randomize Event Date (Jan-Mar Season)
    base_month = random.randint(1, 3)
    base_day = random.randint(1, 28)
    event_date = datetime(2024, base_month, base_day, 6, 0) # 6 AM Start
    
    print(f"Generating Alpha Thesis for event on: {event_date.strftime('%Y-%m-%d')}")
    
    # 2. Setup Timeline (36 Hours)
    hours = np.linspace(0, 36, 72) # 30 min steps
    times = [event_date + timedelta(hours=float(h)) for h in hours]
    
    # 3. Initialize Feeds
    cv_engine = TrafficVision()
    
    # buffers
    market_price = []
    shrink_accum = []
    cv_congestion = []
    
    # Simulation State
    base_price = 170.0 + random.uniform(-5, 5)
    current_delay = 0.0
    
    # Event Trigger: Snow Storm hits at Hour 10
    storm_start = 10.0
    storm_peak = 18.0
    market_reaction_delay = 6.0 # Market is slow
    
    # PORTFOLIO STATE
    initial_capital = 5000.0
    cash = initial_capital
    position = 0 # Contracts
    portfolio_value = []
    
    trade_executed = False
    entry_price = 0.0
    
    for h in hours:
        # A. Traffic Vision Signal (Leading Indicator)
        # Drops speed before storm officially peaks
        if h > storm_start - 2: # Pre-storm slowdown
            # Simulate CV Feed
            feed = cv_engine.analyze_feed() 
            # Inject storm bias
            congestion = feed['metrics']['congestion_index'] + min((h - storm_start)*0.1, 0.4)
            congestion = min(congestion, 1.0)
        else:
            congestion = random.uniform(0.1, 0.3)
        cv_congestion.append(congestion)
        
        # B. Biological Shrink (The Alpha)
        if h > storm_start:
            current_delay += 0.5 + (congestion * 0.2) # Congestion adds to delay
            temp = 10.0 - (h - storm_start) # Getting colder
        else:
            temp = 30.0
            
        shrink_loss = BiologicalDecay.calculate_cattle_shrink(current_delay, temp)
        shrink_accum.append(shrink_loss * 100) # %
        
        # C. Market Price (Lagging)
        # Market reacts at Storm Start + Delay
        reaction_time = storm_start + market_reaction_delay
        
        price_noise = np.random.normal(0, 0.15)
        if h < reaction_time:
            # Flat / random
            price = base_price + price_noise
        else:
            # Pricing in the shrink
            # Price rises as supply drops
            rally_strength = (h - reaction_time) * 0.5
            price = base_price + rally_strength + price_noise
            
        market_price.append(price)
        
        # D. TRADING STRATEGY (The $5000 Growth)
        # Buy Signal: When CV Congestion > 0.6 AND Market hasn't moved.
        current_val = cash
        
        if not trade_executed:
            if congestion > 0.6 and h < reaction_time:
                # BUY CALLS
                # Leverage: 50x (Futures Options)
                # We simply simulate PnL based on price delta
                position = initial_capital / base_price # "Units" (simplified)
                entry_price = price
                trade_executed = True
                # print(f"Trade ENTERED at Hour {h:.1f} @ ${price:.2f}")
        else:
            # Mark to Market
            pnl_pct = (price - entry_price) / entry_price
            # Leverage Logic: 1% move = 20% gain (20x leverage for conservative futures opts)
            leveraged_pnl = pnl_pct * 20.0 
            current_val = initial_capital * (1 + leveraged_pnl)
            
        portfolio_value.append(current_val)

    # 4. Visualization
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0f172a')
    ax1.set_facecolor('#0f172a')
    ax3.set_facecolor('#0f172a')
    
    # SUBPLOT 1: The Thesis (Signals vs Price)
    # Left Axis: Shrink (Red) & CV Congestion (Yellow)
    color_red = '#ef4444'
    ax1.set_xlabel('Event Timeline', color='white')
    ax1.set_ylabel('Bio-Shrink %', color=color_red, fontweight='bold')
    l1, = ax1.plot(times, shrink_accum, color=color_red, linewidth=3, label='Bio-Shrink (Hidden)')
    ax1.tick_params(axis='y', labelcolor=color_red)
    
    # CV Signal (Yellow Overlay)
    color_yellow = '#fbbf24'
    ax1b = ax1.twinx() # Same side? No, maybe secondary signal
    # Let's put CV on the same axis but scaled? No, separate axis is cluttered.
    # Let's just plot CV as a dashed line
    l2, = ax1.plot(times, [c*10 for c in cv_congestion], color=color_yellow, linestyle=':', label='CV Stress Index (x10)')
    
    # Right Axis: Market Price (Green)
    ax2 = ax1.twinx()
    color_green = '#4ade80'
    ax2.set_ylabel('Futures Price ($)', color=color_green, fontweight='bold')
    l3, = ax2.plot(times, market_price, color=color_green, linewidth=2, label='Market Price')
    ax2.tick_params(axis='y', labelcolor=color_green)
    
    # Legend
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    ax1.set_title(f"AgriFlow Arbitrage Event: {event_date.strftime('%Y-%m-%d')}", color='white', fontsize=14)
    
    # Annotate Latency
    idx_trigger = int(storm_start * 2)
    idx_reaction = int((storm_start + market_reaction_delay) * 2)
    
    if idx_reaction < len(times):
        ax1.axvspan(times[idx_trigger], times[idx_reaction], color='yellow', alpha=0.1)
        ax1.text(times[int((idx_trigger+idx_reaction)/2)], max(shrink_accum)/2, "THE ALPHA WINDOW", 
                 color='yellow', ha='center', fontsize=10, fontweight='bold')

    # SUBPLOT 2: The Money ($5k Growth)
    color_blue = '#38bdf8'
    ax3.set_ylabel('Portfolio Value ($)', color=color_blue, fontweight='bold')
    ax3.plot(times, portfolio_value, color=color_blue, linewidth=2)
    ax3.fill_between(times, initial_capital, portfolio_value, color=color_blue, alpha=0.2)
    ax3.axhline(initial_capital, color='white', linestyle='--', alpha=0.5)
    
    final_val = portfolio_value[-1]
    profit = final_val - initial_capital
    roi = (profit / initial_capital) * 100
    
    ax3.text(times[-1], final_val, f"${final_val:,.0f} ({roi:+.1f}%)", color=color_blue, fontweight='bold')
    ax3.set_title("Strategy Performance (20x Lev)", color='white', fontsize=12)
    
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.tight_layout()
    plt.savefig('AgriFlow_Alpha_Thesis.png')
    print(f"Chart Generated: AgriFlow_Alpha_Thesis.png | ROI: {roi:.1f}%")

if __name__ == "__main__":
    generate_dynamic_alpha_thesis()
