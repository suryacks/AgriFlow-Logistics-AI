
import random
import time

class TrafficVision:
    """
    AgriFlow Computer Vision Module.
    Simulates real-time processing of traffic camera feeds to detect:
    1. Commercial Truck Volume (Freight Density)
    2. Average Flow Speed (Optical Flow)
    3. Congestion Events (Stopped Vehicles)
    
    In Production: Connects to state DOT RTSP streams (YOLOv8 Inference).
    In Demo: Simulates realistic inference outputs.
    """
    
    def __init__(self):
        self.cameras = [
            "I-80_DesMoines_East",
            "I-29_Omaha_North",
            "I-80_Chicago_Approach",
            "I-70_KansasCity_West"
        ]
        
    def analyze_feed(self, camera_id=None):
        if not camera_id:
            camera_id = random.choice(self.cameras)
            
        # Simulate processing time (CV latency)
        # time.sleep(0.05) 
        
        # Stochastic Simulation based on time of day?
        # For Alpha Thesis, we want "Signal".
        
        # Generate raw detection data
        truck_count = random.randint(5, 45)
        car_count = random.randint(20, 150)
        
        # Optical Flow Speed (mph)
        # If density is high, speed drops
        density = (truck_count * 3 + car_count) / 200.0 # Normalized 0-1
        base_speed = 70.0
        avg_speed = max(5.0, base_speed * (1.0 - (density**2)))
        
        # Add noise
        avg_speed += random.uniform(-5, 5)
        
        return {
            "camera_id": camera_id,
            "timestamp": time.time(),
            "detections": {
                "commercial_trucks": truck_count,
                "passenger_vehicles": car_count,
                "total_objects": truck_count + car_count
            },
            "metrics": {
                "optical_flow_speed_mph": round(avg_speed, 1),
                "congestion_index": round(density, 2),
                "lane_occupancy_pct": round(density * 80, 1)
            },
            "status": "ONLINE"
        }
