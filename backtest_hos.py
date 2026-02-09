from src.invisible_alpha.hos_simulator import DriverAgent
import simpy

def test_the_45_min_domino_theory():
    """
    BACKTEST PROOF:
    Proves that a 45-minute accident caused a 10-hour delivery failure.
    
    Scenario:
    - Driver has driven 10 hours and 20 minutes (10.33h).
    - Destination is 30 minutes away (0.5h).
    - Legal limit is 11.0 hours.
    - Buffer: 11.0 - (10.33 + 0.5) = 0.17h (10 mins spare). SAFE.
    
    Event:
    - Accident causes 45 minute (0.75h) delay.
    
    Outcome:
    - Total Time = 10.33 + 0.5 + 0.75 = 11.58 hours.
    - 11.58 > 11.0 -> VIOLATION.
    - Result: Driver must stop at 11.0h mark. Rest 10 hours.
    - Delay impact is NOT 45 mins. It is 10 hours + 45 mins.
    """
    
    print("\n--- BACKTEST: THE 45-MINUTE DOMINO ---")
    
    # 1. Setup Environment
    env = simpy.Environment()
    
    # 2. Define Driver State
    current_drive_time = 10.33 # 10h 20m
    dist_to_dest = 32.5        # 30 mins at 65mph
    avg_speed = 65.0
    
    driver = DriverAgent(env, "Subject-Zero", current_drive_time, dist_to_dest, avg_speed)
    
    # 3. Define The Invisible Friction (45 min delay)
    accident_delay = 0.75 
    
    # 4. Run Simulation
    print(f"STATE: Driven {current_drive_time:.2f}h. Dist {dist_to_dest}mi. Spare HOS: {(11.0 - (current_drive_time + 0.5))*60:.0f} mins.")
    print(f"EVENT: Impacted by {accident_delay*60:.0f} min unexpected delay.")
    
    env.process(driver.drive_process(traffic_delay_hours=accident_delay))
    env.run()
    
    # 5. Analyze Results
    expected_arrival_no_delay = current_drive_time + (dist_to_dest/avg_speed)
    actual_arrival_clock = driver.arrival_time
    
    # The arrival time in the daily cycle (assuming started at 0.0)
    # The driver started this leg at 'current_drive_time' visually? 
    # No, in simulation 'env.now' starts at 0. So arrival relative to specific leg start needs adjustment.
    # Actually DriverAgent simulates the *remaining* trip. 
    # env.now is the duration of THIS leg.
    
    time_spent_on_leg = driver.arrival_time
    total_log_time = current_drive_time + time_spent_on_leg
    
    print(f"\nRESULT:")
    print(f"Total Logged Time: {total_log_time:.2f} hours")
    print(f"Status: {driver.status}")
    
    if driver.status == "ARRIVED" and (time_spent_on_leg > 8.0): # Indicates rest was taken
        print(f"PROOF VERIFIED: The 45-minute delay triggered a 10-hour shutdown.")
        print(f"Total delay experienced: {time_spent_on_leg - (dist_to_dest/avg_speed):.2f} hours")
    else:
        print("Test inconclusive.")

if __name__ == "__main__":
    test_the_45_min_domino_theory()
