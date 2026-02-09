import simpy
import random

class HOSException(Exception):
    """Raised when Hours of Service are exceeded."""
    pass

class DriverAgent:
    def __init__(self, env, driver_id, current_drive_time, distance_to_dest, avg_speed=65.0):
        self.env = env
        self.driver_id = driver_id
        self.current_drive_time = current_drive_time # Hours
        self.distance_to_dest = distance_to_dest     # Miles
        self.avg_speed = avg_speed
        self.status = "DRIVING"
        self.arrival_time = None
        self.hos_limit = 11.0
        self.rest_period = 10.0

    def drive_process(self, traffic_delay_hours=0.0):
        """
        Simulates the drive. 
        traffic_delay_hours: Unexpected delay added to the trip.
        """
        # Calculate base time needed
        time_to_dest = self.distance_to_dest / self.avg_speed
        
        # Projected total drive time
        total_drive_time = self.current_drive_time + time_to_dest + traffic_delay_hours
        
        # Check for HOS Violation logic
        # If the delay pushes them over 11 hours, they MUST stop for 10 hours.
        if total_drive_time > self.hos_limit:
            # Determine when the violation occurs (at the 11th hour mark)
            # Actually, in real life, if they can't make it, they stop BEFORE 11. 
            # Or if they get stuck in traffic and cross 11, they are in violation and must stop immediately.
            
            # For this simulation: The "Domino Effect"
            # The delay makes the trip impossible within the legal window.
            
            remaining_legal_time = self.hos_limit - self.current_drive_time
            
            # Drive until limit
            yield self.env.timeout(remaining_legal_time)
            
            self.status = "HOS_FORCED_REST"
            print(f"[{self.env.now:.2f}h] Driver {self.driver_id} HIT HOS LIMIT! Forced 10h rest.")
            
            # The 10-hour shutdown
            yield self.env.timeout(self.rest_period)
            
            self.status = "RESUMING"
            # Finish the drive (remaining distance)
            # Distance covered so far: (remaining_legal_time - traffic_delay_hours(if applicable)) * speed
            # Simply: we have (total_drive_time - 11.0) hours left to drive
            remaining_drive = total_drive_time - self.hos_limit
            yield self.env.timeout(remaining_drive)
            
        else:
            # No violation, just drive
            yield self.env.timeout(time_to_dest + traffic_delay_hours)

        self.status = "ARRIVED"
        self.arrival_time = self.env.now
        print(f"[{self.env.now:.2f}h] Driver {self.driver_id} Arrived.")

def run_fleet_simulation(fleet_size=100, traffic_delay_distribution=lambda: random.expovariate(1.0/0.5)):
    """
    Simulates a fleet where drivers have varying remaining drive times.
    traffic_delay_distribution: function returning delay in hours.
    """
    env = simpy.Environment()
    drivers = []
    results = {"total": 0, "hos_failures": 0, "delays": []}
    
    for i in range(fleet_size):
        # Setup random state: Driver has been driving between 6 and 10 hours already
        current_time = random.uniform(6.0, 10.5) 
        dist = random.uniform(50, 200) # Miles left
        
        driver = DriverAgent(env, f"D-{i}", current_time, dist)
        drivers.append(driver)
        
        # Inject traffic delay?
        # Let's say 20% of fleet hits an "Invisible Friction" event (accident)
        delay = 0.0
        if random.random() < 0.20:
            delay = traffic_delay_distribution() # e.g. 0.75 hours (45 mins)
            
        env.process(driver.drive_process(traffic_delay_hours=delay))
        results["delays"].append((driver, delay))

    env.run()
    
    # Calculate stats
    for d, delay in results["delays"]:
        # If arrival time implies a rest was taken (e.g., took longer than dist/speed + delay + buffer)
        expected_time = (d.distance_to_dest / d.avg_speed) + delay
        if d.arrival_time > expected_time + 9.0: # 9.0 buffer clearly indicates the 10h rest
            results["hos_failures"] += 1
            
    results["total"] = fleet_size
    results["failure_rate"] = results["hos_failures"] / fleet_size
    return results

if __name__ == "__main__":
    # Test Run
    print("--- HOS DOMINO SIMULATION ---")
    res = run_fleet_simulation()
    print(f"Fleet Size: {res['total']}")
    print(f"HOS Failures (Domino Effect): {res['hos_failures']}")
    print(f"HOS Failure Rate: {res['failure_rate']*100:.1f}%")
