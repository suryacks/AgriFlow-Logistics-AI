
import gymnasium as gym
import numpy as np
import networkx as nx
import torch
from gymnasium import spaces
import random

class AgriFlowFleetEnv(gym.Env):
    def __init__(self, graph: nx.DiGraph, num_trucks=25, enable_traffic=True):
        super(AgriFlowFleetEnv, self).__init__()
        self.graph = graph
        self.supply_nodes = sorted([n for n, d in graph.nodes(data=True) if d.get('type') == 'supply'])
        self.demand_nodes = sorted([n for n, d in graph.nodes(data=True) if d.get('type') == 'demand'])
        
        self.n_supply = len(self.supply_nodes)
        self.n_demand = len(self.demand_nodes)
        self.num_trucks = num_trucks
        self.enable_traffic = enable_traffic
        
        # --- FLEET MANAGEMENT ---
        # Truck State: [Current_Node_Index (int), Is_Loaded (bool), Load_Amount (float)]
        # We also track exact string ID of location internally
        self.trucks = [] 
        
        # --- ACTION SPACE ---
        # The user requested: Select Truck, Source, Destination.
        # This is combinatorially huge (5 * 180 * 548).
        # We simplify: The Agent is called when a Truck is IDLE.
        # The Agent selects: (Source_Index, Destination_Index).
        # Truck is implicit (the idle one).
        # Source Index: 0 to 179. Destination Index: 0 to 547.
        # We use MultiDiscrete to separate them.
        self.action_space = spaces.MultiDiscrete([self.n_supply, self.n_demand])
        
        # --- OBSERVATION SPACE ---
        # 1. Truck Locations (One-hot encoded? Too big. Normalized Lat/Lon or Node Index?) -> Node Index / Total Nodes
        # 2. Supply Levels (Normalized)
        # 3. Demand Backlog (Normalized)
        # 4. Spoilage Risk / Heat Factor (Scalar global or per route? Global for simplicity)
        
        # Size: (Num_Trucks) + (Num_Supply) + (Num_Demand) + (1 Heat Factor) + (10 Traffic)
        self.obs_dim = self.num_trucks + self.n_supply + self.n_demand + 1 + 10
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_dim,), dtype=np.float32)
        
        self.max_milk = 100000.0
        
        # Exploration Tracking
        self.visited_customers = set()

    def reset(self, seed=None, heat_start=0.5, traffic_prob=0.1):
        super().reset(seed=seed)
        self.visited_customers = set()
        self.traffic_prob = traffic_prob
        
        # 1. Reset Inventory
        self.current_supply = {}
        for n in self.supply_nodes:
            self.current_supply[n] = self.graph.nodes[n]['data'].available_qty
            
        self.current_demand = {}
        for n in self.demand_nodes:
            self.current_demand[n] = self.graph.nodes[n]['data'].daily_capacity
            
        # 2. Reset Fleet
        # Start all trucks at random Supply Nodes
        self.trucks = []
        for i in range(self.num_trucks):
            start_node_idx = random.randint(0, self.n_supply - 1)
            self.trucks.append({
                'id': i,
                'loc_idx': start_node_idx, 
                'is_at_supply': True,
                'load': 0.0
            })
            
        # 3. Env Factors
        self.heat_factor = heat_start 
        
        # 4. Traffic
        self.traffic_events = {} 
        if self.enable_traffic and random.random() < self.traffic_prob:
             self._generate_traffic_jams()
        
        self.active_truck_idx = 0 
        self.step_count = 0
        
        return self._get_state(), {}

    def _generate_traffic_jams(self):
        # Clear old
        self.traffic_events = {}
        # Pick 10 random potential jams
        if self.graph.number_of_edges() > 0:
            all_edges = list(self.graph.edges())
            jammed_edges = random.sample(all_edges, min(10, len(all_edges)))
            for u, v in jammed_edges:
                self.traffic_events[(u, v)] = 10.0 # 10x cost/time!

    def _get_state(self):
        # 1. Truck Locs (Normalized by max range of indices)
        # We simply output the index normalized
        truck_state = []
        max_idx = self.n_supply + self.n_demand
        for t in self.trucks:
            # Map location to a single float
            val = t['loc_idx'] / max_idx if t['is_at_supply'] else (t['loc_idx'] + self.n_supply) / max_idx
            truck_state.append(val)
            
        # 2. Supply
        s_vec = [self.current_supply[n] / self.max_milk for n in self.supply_nodes]
        
        # 3. Demand
        d_vec = [self.current_demand[n] / self.max_milk for n in self.demand_nodes]
        
        # 4. Traffic Sensors (Do we see the jams?)
        # For simplicity, we pass a boolean "Jam Alert" vector for the top jams?
        # Or just append a scalar "Global Traffic Risk".
        # Let's append 10 indicators for the 10 potential jam spots (Static mapping)
        # This is strictly for the AI to "see" the jam.
        # Ideally, we embed graph features, but for MLP, we'll append a "Chaos Vector".
        traffic_vec = [1.0] * 10 # Placeholder for actual active jams. 
        # (In a real system, we'd feed real Waze data).
        
        state = np.array(truck_state + s_vec + d_vec + [self.heat_factor] + traffic_vec, dtype=np.float32)
        return state

    def get_source_mask(self):
        # Mask Sources with Milk (> 100)
        mask = np.zeros(self.n_supply, dtype=np.float32)
        for i, n in enumerate(self.supply_nodes):
            if self.current_supply[n] > 100:
                mask[i] = 1.0
        if mask.sum() == 0: mask[:] = 1.0
        return torch.tensor(mask, dtype=torch.bool)

    def get_dest_mask(self, source_idx):
        # Mask Dests that are CONNECTED to source_idx AND Need Milk
        source_id = self.supply_nodes[source_idx]
        mask = np.zeros(self.n_demand, dtype=np.float32)
        
        # Determine valid neighbors
        if self.graph.has_node(source_id):
            neighbors = list(self.graph.successors(source_id))
            for i, d_node in enumerate(self.demand_nodes):
                if d_node in neighbors and self.current_demand[d_node] > 100:
                    mask[i] = 1.0
                    
        if mask.sum() == 0: 
            # If no connected destinations need milk, allow ANY connected destination (to dump/wait) or just fail
            # Fallback: Allow any connected
             for i, d_node in enumerate(self.demand_nodes):
                 if d_node in neighbors: mask[i] = 1.0
                 
        if mask.sum() == 0: mask[:] = 1.0 # Last resort (will punish)
            
        return torch.tensor(mask, dtype=torch.bool)

    def step(self, action_tuple):
        # SWARM MARL CHANGE:
        source_idx, dest_idx = action_tuple
        truck = self.trucks[self.active_truck_idx]
        
        source_id = self.supply_nodes[source_idx]
        dest_id = self.demand_nodes[dest_idx]
        
        reward = 0
        done = False
        
        has_route = self.graph.has_edge(source_id, dest_id)
        
        if has_route:
            edge = self.graph[source_id][dest_id]
            base_dist = edge.get('distance', 50.0)
            
            # --- TRAFFIC LOGIC ---
            # Is this edge jammed?
            traffic_mult = self.traffic_events.get((source_id, dest_id), 1.0)
            
            # Distance/Time is multiplied by Traffic
            dist = base_dist * traffic_mult
            
            # Relocation
            curr_loc = self.supply_nodes[truck['loc_idx']] if truck['is_at_supply'] else self.demand_nodes[truck['loc_idx']]
            reloc_dist = 50.0 if curr_loc != source_id else 0.0
            
            total_dist = dist + reloc_dist
            
            # Spoilage Risk SCALES with Traffic Time
            risk = (total_dist / 200.0) * (1 + self.heat_factor**2)
            
            avail = self.current_supply[source_id]
            need = self.current_demand[dest_id]
            moved = min(avail, need, 20000.0)
            
            loss_pct = min(risk, 1.0)
            delivered = moved * (1.0 - loss_pct)
            spoiled = moved * loss_pct
            
            self.current_supply[source_id] -= moved
            self.current_demand[dest_id] -= delivered
            
            truck['loc_idx'] = dest_idx
            truck['is_at_supply'] = False
            
            # Reward
            revenue = delivered * 0.50
            cost = total_dist * 1.50 
            penalty = spoiled * 2.0
            profit = revenue - cost - penalty
            
            reward += profit
            
            # EXPLORATION BONUS
            if dest_id not in self.visited_customers:
                reward += 50.0 
                self.visited_customers.add(dest_id)
            
            if moved < 100: reward -= 50.0 
            
        else:
            reward -= 500.0 
            
        self.step_count += 1
        
        # Weather & Traffic Dynamics
        if self.step_count % self.num_trucks == 0:
             self.heat_factor = min(max(self.heat_factor + np.random.normal(0, 0.05), 0.0), 1.0)
             # Randomly clear or create jams
             if self.enable_traffic and random.random() < self.traffic_prob: self._generate_traffic_jams()
             
        self.active_truck_idx = (self.active_truck_idx + 1) % self.num_trucks
        
        if self.step_count >= 1000: 
            done = True
            
        return self._get_state(), reward, done, False, {}
