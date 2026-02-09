import os
# import googlemaps
from datetime import datetime

class TrafficAnomalyDetector:
    def __init__(self, api_key=None):
        self.api_key = api_key
        # self.gmaps = googlemaps.Client(key=api_key) if api_key else None
        
        # I-80 / I-90 Key Segments (Lat/Lng) - Meat/Dairy Corridors
        self.corridors = {
            "I-80_NE": {"start": "40.9, -98.3", "end": "40.8, -99.1"}, # Nebraska (Beef)
            "I-90_WI": {"start": "43.1, -89.3", "end": "43.9, -90.1"}, # Wisconsin (Dairy)
        }

    def get_weather_mock(self, location):
        """
        Mock weather service. Real impl would use Open-Meteo or similar.
        Returns: 'Clear', 'Rain', 'Snow', 'Fog'
        """
        # For demo purposes, we usually return Clear to isolate non-weather friction
        return "Clear"

    def check_segment(self, segment_id):
        """
        Checks a segment for 'Invisible Friction'.
        Logic: Speed < 50% AND Weather == Clear.
        """
        coords = self.corridors.get(segment_id)
        if not coords:
            return None

        # --- MOCK DATA FETCH (No API Key) ---
        # unique logic: 
        # 1. Fetch real-time traffic (duration_in_traffic)
        # 2. Fetch Free Flow duration
        # 3. Fetch Weather
        
        # MOCK Response representing a Construction Accident in Clear Weather
        traffic_data = {
            "distance_meters": 80000, # 80 km
            "duration_free_flow": 3600, # 60 mins -> 80km/h
            "duration_in_traffic": 7800, # 130 mins -> ~37km/h (Massive slowdown)
        }
        
        weather = self.get_weather_mock(coords['start'])
        
        # ANALYSIS
        free_flow_speed = traffic_data['distance_meters'] / traffic_data['duration_free_flow']
        current_speed = traffic_data['distance_meters'] / traffic_data['duration_in_traffic']
        
        ratio = current_speed / free_flow_speed
        
        is_friction = False
        reason = "Normal"
        
        if ratio < 0.5:
            if weather == "Clear":
                is_friction = True
                reason = "INVISIBLE_FRICTION: Severe slowdown in CLEAR weather (Accident/Construction?)"
            else:
                reason = f"Weather-related slowdown ({weather})"
        
        return {
            "segment": segment_id,
            "current_speed_mph": round(current_speed * 2.23694, 1), # m/s to mph
            "ratio": round(ratio, 2),
            "weather": weather,
            "anomaly_detected": is_friction,
            "reason": reason
        }

    def scan_all_corridors(self):
        results = []
        for segment in self.corridors:
            results.append(self.check_segment(segment))
        return results

if __name__ == "__main__":
    detector = TrafficAnomalyDetector()
    alerts = detector.scan_all_corridors()
    for alert in alerts:
        if alert['anomaly_detected']:
            print(f"🚨 ALERTS: {alert['segment']} - {alert['reason']}")
            print(f"   Speed: {alert['current_speed_mph']} mph (Ratio: {alert['ratio']})")
