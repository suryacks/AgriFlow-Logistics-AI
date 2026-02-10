
import math

class BiologicalDecay:
    """
    Bio-Financial Arbitrage Engine.
    Calculates the 'Invisible Cost' of transport delays on biological assets.
    
    Based on:
    1. Oklahoma State University Extension: "Shrink in Beef Cattle: A Marketing Consideration"
    2. Journal of Dairy Science: "Thermal Load and bacterial growth in raw milk transport"
    """
    
    @staticmethod
    def calculate_cattle_shrink(hours_delay, temp_f=70):
        """
        Calculates weight loss (shrink) during transport delays.
        
        Formula:
        Base Shrink = 2% (Standard Transport)
        Extra Shrink = 0.1% per hour of delay > 4 hours.
        Thermal Multiplier: Increases by 10% for every 10F above 80F.
        
        Args:
            hours_delay (float): Hours the truck is stopped/delayed beyond schedule.
            temp_f (float): Ambient Temperature (Fahrenheit).
            
        Returns:
            float: Shrink Percentage (e.g., 0.035 for 3.5%)
        """
        base_shrink = 0.02
        
        if hours_delay <= 0:
            return base_shrink
            
        # Linear Decay model from OSU Extension
        # "Cattle shrink approximately 1% of body weight for every 4 hours without feed/water."
        extra_shrink = (hours_delay / 4.0) * 0.01
        
        # Thermal Stress Multiplier
        heat_stress = 1.0
        if temp_f > 80:
            heat_stress += (temp_f - 80) * 0.01 # 1% increase per degree? No, 10% per 10 deg -> 0.01 per 1.
            
        total_shrink = base_shrink + (extra_shrink * heat_stress)
        return min(total_shrink, 0.10) # Cap at 10% (Critcal failure)

    @staticmethod
    def calculate_milk_spoilage(hours_delay, external_temp_c):
        """
        Calculates Probability of Spoilage (Bacterial Bloom Risk).
        
        Based on exponential growth of psychrotrophic bacteria in raw milk 
        when tanker insulation fails linearly over time.
        """
        # Tanker allows internal temp to rise 0.5C per hour of delay if AC fails or static
        internal_temp = 4.0 + (hours_delay * 0.5)
        
        if internal_temp < 7.0:
            return 0.0 # Safe Zone
            
        # Exponential Growth Risk (Gompertz Function approximation)
        # Risk returns 0.0 - 1.0
        risk = 1.0 - math.exp(-0.5 * (internal_temp - 7.0))
        return max(0.0, min(risk, 1.0))
