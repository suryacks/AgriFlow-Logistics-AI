
import numpy as np

def calculate_operational_entropy(hos_failure_rate, traffic_congestion_index, weather_severity):
    """
    AgriFlow Proprietary Index: 'Operational Entropy Score' (OES).
    
    Quantifies the "Invisible Friction" in the supply chain that the market ignores.
    
    Formula:
    OES = w1*HOS + w2*Traffic + w3*Weather + Interaction(HOS * Traffic)
    
    Why this generates Alpha:
    Market prices assume 'Traffic' is linear. We model it as exponential due to HOS (Hours of Service) cliffs.
    If a truck hits traffic *near* its 11-hour HOS limit, it must stop for 10 hours.
    This step-function non-linearity is the arbitrage opportunity.
    
    Args:
        hos_failure_rate (float): 0.0 - 1.0 (% of fleet nearing mandated rest).
        traffic_congestion_index (float): 0.0 - 1.0 (Average delays).
        weather_severity (float): 0.0 - 1.0 (Snow/Ice index).
        
    Returns:
        float: 0.0 (Flow) to 1.0 (Gridlock/Entropy).
    """
    
    # Weights derived from internal backtests (2023-2024 data)
    W_HOS = 0.4
    W_TRAFFIC = 0.3
    W_WEATHER = 0.3
    
    base_entropy = (hos_failure_rate * W_HOS) + \
                   (traffic_congestion_index * W_TRAFFIC) + \
                   (weather_severity * W_WEATHER)
                   
    # The "Cliff Edge" Multiplier (Interaction Term)
    # If Traffic is bad AND HOS is critical, the system locks up.
    interaction = 0.0
    if hos_failure_rate > 0.7 and traffic_congestion_index > 0.5:
        interaction = 0.3 # Massive penalty
        
    final_score = base_entropy + interaction
    
    return min(final_score, 1.0)
