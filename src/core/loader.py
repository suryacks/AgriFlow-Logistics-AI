
import pandas as pd
import os
import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

@dataclass
class SupplyNode:
    id: str
    location_name: str
    available_qty: float
    start_time: str = "00:00:00"

@dataclass
class DemandNode:
    id: str
    location_name: str
    daily_capacity: float
    region: str

class ScenarioLoader:
    def __init__(self):
        self.supply_nodes = {}
        self.demand_nodes = {}
        self.edges = []
        self.graph = nx.DiGraph()

    def load_scenario(self):
        print("--- LOADING LOGISTICS SCENARIO ---")
        
        # 1. Load Demand (Sinks)
        demand_file = os.path.join(DATA_DIR, "Location Receving Capacity.XLSX")
        if os.path.exists(demand_file):
            df_dem = pd.read_excel(demand_file)
            # Expected cols: Location, Location Description, Daily Capacity
            for _, row in df_dem.iterrows():
                loc_id = str(row.get('Location', 'Unknown'))
                node = DemandNode(
                    id=loc_id,
                    location_name=row.get('Location Description', 'Unknown'),
                    daily_capacity=max(float(row.get('Daily Capacity', 0) if not pd.isna(row.get('Daily Capacity')) else 0), 20000.0), # Min 20k
                    region=row.get('Region', 'Unknown')
                )
                self.demand_nodes[loc_id] = node
                self.graph.add_node(loc_id, type='demand', data=node)
            print(f"Loaded {len(self.demand_nodes)} Demand Locations.")

        # 2. Load Supply (Sources)
        supply_file = os.path.join(DATA_DIR, "Milk Availability at Route.XLSX")
        if os.path.exists(supply_file):
            df_sup = pd.read_excel(supply_file)
            # Expected cols: Source Location, Confirmed Quantity (ATP)
            # We group by Source Location to get total available
            # Note: The file seems to have multiple rows per location (time slots).
            # For simplicity V1, we sum them up or take the max availability.
            
            # Using 'Source Location' as key
            if 'Source Location' in df_sup.columns:
                grouped = df_sup.groupby('Source Location')['Confirmed Quantity (ATP)'].sum()
                for loc_id, qty in grouped.items():
                    loc_id = str(loc_id)
                    node = SupplyNode(
                        id=loc_id,
                        location_name=f"Farm_{loc_id}", # Name might not be separate
                        available_qty=max(float(qty), 20000.0),
                        start_time="00:00:00" # Placeholder
                    )
                    # Hack: Store max_hold_time on the object dynamically since Dataclass is frozen-ish or I define it
                    # Let's update the SupplyNode definition first or just add it as attribute dynamically in python
                    node.max_hold_time = float(row.get('Max Hold Time', 72.0) if not pd.isna(row.get('Max Hold Time')) else 72.0)
                    
                    self.supply_nodes[loc_id] = node
                    self.graph.add_node(loc_id, type='supply', data=node)
                print(f"Loaded {len(self.supply_nodes)} Supply Routes.")
            else:
                print("Error: 'Source Location' column missing in supply file.")

        # 3. Load Transportation Lanes (Edges)
        lanes_file = os.path.join(DATA_DIR, "Transportation lane from Route to Location.xlsx")
        if os.path.exists(lanes_file):
            df_lane = pd.read_excel(lanes_file)
            # Expected: Start Location, Destination Location, Transportation Cost, Transportation Distance(Miles)
            count = 0
            for _, row in df_lane.iterrows():
                u = str(row.get('Start Location ', 'Unknown')).strip() # Note the space in col name from inspection
                v = str(row.get('Destination Location', 'Unknown')).strip()
                
                if u in self.supply_nodes and v in self.demand_nodes:
                    cost = float(row.get('Transportation Cost', 10.0))
                    dist = float(row.get('Transportation Distance(Miles)', 50.0))
                    duration_str = str(row.get('Transportation Duration(Hrs)', '01:00:00'))
                    
                    # Convert duration string to hours float (approx)
                    # Assuming HH:MM:SS
                    try:
                        h, m, s = map(int, duration_str.split(':'))
                        duration = h + m/60 + s/3600
                    except:
                        duration = 1.0 # Default
                    
                    self.graph.add_edge(u, v, cost=cost, distance=dist, duration=duration)
                    count += 1
            print(f"Loaded {count} Transportation Lanes (Valid Edges).")
        
        # Estimate needed trucks
        total_supply_vol = sum(n.available_qty for n in self.supply_nodes.values())
        # Assumption: Truck does 2 trips/day, carries 20k
        needed_trucks = int(total_supply_vol / (20000 * 2)) + 5 # Buffer
        print(f"Estimated Fleet Size needed: {needed_trucks} Trucks (based on {total_supply_vol:,.0f} lbs/day)")
        self.suggested_fleet_size = max(needed_trucks, 5)
        
        return self.graph, self.suggested_fleet_size

if __name__ == "__main__":
    loader = ScenarioLoader()
    G = loader.load_scenario()
    print(f"Graph Construction Complete: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges.")
