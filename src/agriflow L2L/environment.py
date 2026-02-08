import gymnasium as gym
import numpy as np
import osmnx as ox
import networkx as nx
import math
import os
import random
from config import *

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000 
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

class AgriFlowEnv(gym.Env):
    def __init__(self):
        super(AgriFlowEnv, self).__init__()
        
        # 1. Load Map
        if os.path.exists(GRAPH_FILENAME):
            self.G = ox.load_graphml(GRAPH_FILENAME)
        else:
            print("Initializing AgriFlow Graph...")
            start_loc = ox.geocode(DEFAULT_START)
            self.G = ox.graph_from_point(start_loc, dist=6000, network_type='drive')
            self.G = ox.add_edge_speeds(self.G)
            self.G = ox.add_edge_travel_times(self.G)
            self.G = ox.truncate.largest_component(self.G, strongly=True)
            ox.save_graphml(self.G, GRAPH_FILENAME)

        self.nodes_list = list(self.G.nodes())
        
        # Initialize with Default Mission
        self.set_mission_addresses(DEFAULT_START, DEFAULT_END)

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(15,), dtype=np.float32)

    def set_mission_addresses(self, start_addr, end_addr):
        """Sets a specific mission from addresses."""
        s = ox.geocode(start_addr)
        e = ox.geocode(end_addr)
        self.start_node = ox.distance.nearest_nodes(self.G, s[1], s[0])
        self.target_node = ox.distance.nearest_nodes(self.G, e[1], e[0])
        self._update_heuristic()

    def set_random_mission(self):
        """Sets a random start and end point for testing generalization."""
        self.start_node = random.choice(self.nodes_list)
        self.target_node = random.choice(self.nodes_list)
        # Ensure they aren't the same
        while self.start_node == self.target_node:
            self.target_node = random.choice(self.nodes_list)
        self._update_heuristic()

    def _update_heuristic(self):
        """Pre-computes the 'Google Maps' baseline (Dijkstra) for the current target."""
        try:
            self.global_heuristic = nx.shortest_path_length(
                self.G, target=self.target_node, weight='travel_time'
            )
        except nx.NetworkXNoPath:
            self.global_heuristic = {}

    def get_neighbor_features(self, neighbor):
        # Feature 1: Deterministic Cost (Normalized)
        h_cost = self.global_heuristic.get(neighbor, 10000) / 3600.0
        
        # Feature 2: Speed
        edge_data = self.G.get_edge_data(self.current_node, neighbor)
        edge = edge_data[min(edge_data.keys())]
        speed = edge.get('speed_kph', 30)
        if isinstance(speed, list): speed = speed[0]
        speed = float(speed) / 100.0
        
        # Feature 3: Bearing Deviation
        curr_d = self.G.nodes[self.current_node]
        neigh_d = self.G.nodes[neighbor]
        tgt_d = self.G.nodes[self.target_node]
        
        bearing_goal = ox.bearing.calculate_bearing(curr_d['y'], curr_d['x'], tgt_d['y'], tgt_d['x'])
        bearing_road = ox.bearing.calculate_bearing(curr_d['y'], curr_d['x'], neigh_d['y'], neigh_d['x'])
        bearing_diff = abs(bearing_goal - bearing_road) / 180.0
        
        return np.array([h_cost, speed, bearing_diff], dtype=np.float32)

    def step(self, action_idx):
        neighbors = list(self.G.successors(self.current_node))
        neighbors.sort(key=lambda n: self.global_heuristic.get(n, float('inf')))
        
        if action_idx >= len(neighbors):
            next_node = neighbors[0] 
        else:
            next_node = neighbors[action_idx]

        # Calculate Reward (Negative Time)
        edge_data = self.G.get_edge_data(self.current_node, next_node)
        edge = edge_data[min(edge_data.keys())]
        time_cost = edge.get('travel_time', 5.0)
        
        reward = -time_cost
        
        self.current_node = next_node
        terminated = (self.current_node == self.target_node)
        
        if terminated:
            reward += 1000 
            
        return self._get_state_for_neighbors(), reward, terminated, False, {}

    def _get_state_for_neighbors(self):
        neighbors = list(self.G.successors(self.current_node))
        neighbors.sort(key=lambda n: self.global_heuristic.get(n, float('inf')))
        
        state_tensor = np.zeros((5, 3), dtype=np.float32)
        for i, n in enumerate(neighbors[:5]):
            state_tensor[i] = self.get_neighbor_features(n)
        return state_tensor.flatten()

    def reset(self):
        self.current_node = self.start_node
        return self._get_state_for_neighbors()