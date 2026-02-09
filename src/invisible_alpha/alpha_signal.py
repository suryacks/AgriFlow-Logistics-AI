from src.invisible_alpha.hos_simulator import run_fleet_simulation
from src.invisible_alpha.traffic_anomaly import TrafficAnomalyDetector
from src.invisible_alpha.facility_watch import FacilityWatch

class InvisibleAlphaSignal:
    def __init__(self):
        self.traffic = TrafficAnomalyDetector()
        self.facility = FacilityWatch()
        
    def generate_signal(self):
        print("--- GENERATING INVISIBLE ALPHA SIGNAL ---")
        signals = []
        
        # 1. RUN HOS SIMULATION (The Domino Effect)
        # Scenario: We detected a 45 min delay on I-80. How many drivers fail?
        print("[1] Simulating Fleet HOS Impact...")
        # Lambda creates a 45min delay (0.75h) constant for test
        sim_results = run_fleet_simulation(fleet_size=500, traffic_delay_distribution=lambda: 0.75) 
        
        hos_failure_rate = sim_results['failure_rate']
        print(f"    Fleet Impact: {hos_failure_rate*100:.1f}% of drivers triggered 10h forced rest.")
        
        if hos_failure_rate > 0.20:
            signals.append(f"SELL: Live Cattle (Logistics Failure Rate {hos_failure_rate*100:.0f}% > 20%)")
            
        # 2. RUN TRAFFIC ANOMALY
        print("[2] Scanning Corridors for Non-Weather Friction...")
        traffic_alerts = self.traffic.scan_all_corridors()
        for t in traffic_alerts:
            if t['anomaly_detected']:
                 print(f"    Anomaly: {t['segment']} is {t['ratio']}x speed in {t['weather']} weather.")
                 signals.append(f"WARNING: Invisible Friction on {t['segment']}")

        # 3. FACILITY WATCH
        print("[3] Analyzing Driver Sentiment...")
        facility_alerts = self.facility.analyze_sentiment()
        for f in facility_alerts:
             print(f"    Facility: {f['facility']} congestion detected.")
             signals.append(f"SELL: {f['facility']} Supply Chain Bottleneck")

        # FINAL DECISION
        print("\n=== ALPHA DECISION ===")
        if not signals:
            print("NO SIGNAL: Supply Chain Operating Normally.")
            return "HOLD"
        else:
            for s in signals:
                print(f"🔥 {s}")
            return "SELL"

if __name__ == "__main__":
    alpha = InvisibleAlphaSignal()
    alpha.generate_signal()
