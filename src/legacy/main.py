import gymnasium as gym
import numpy as np
import osmnx as ox
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# We use a smaller map so you can see individual road pixels
START_ADDRESS = "1001 Centurion Ln, Vernon Hills, IL 60061"
END_ADDRESS = "200 Marriott Dr, Lincolnshire, IL 60069" # Closer target

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, 'visualizations')
GRAPH_FILE = os.path.join(RESOURCES_DIR, "vernon_micro_graph.graphml")

HIDDEN_SIZE = 128
LEARNING_RATE = 0.001
GAMMA = 0.95
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. THE ENVIRONMENT (The World) ---
class AgriFlowEnv(gym.Env):
    def __init__(self):
        super(AgriFlowEnv, self).__init__()
        
        # 1. Load Map (Micro Scale: 800 meters)
        if os.path.exists(GRAPH_FILE):
            print(f"Loading cached map from {GRAPH_FILE}...")
            self.G = ox.load_graphml(GRAPH_FILE)
        else:
            print("Downloading Micro-Map (800m radius)...")
            start_loc = ox.geocode(START_ADDRESS)
            # dist=800 makes a small, zoomable map
            self.G = ox.graph_from_point(start_loc, dist=800, network_type='drive')
            self.G = ox.add_edge_speeds(self.G)
            self.G = ox.add_edge_travel_times(self.G)
            self.G = ox.truncate.largest_component(self.G, strongly=True)
            ox.save_graphml(self.G, GRAPH_FILE)

        # 2. Setup Start/End Nodes
        start_loc = ox.geocode(START_ADDRESS)
        end_loc = ox.geocode(END_ADDRESS)
        self.start_node = ox.distance.nearest_nodes(self.G, start_loc[1], start_loc[0])
        self.target_node = ox.distance.nearest_nodes(self.G, end_loc[1], end_loc[0])
        
        # Calculate Map Dimensions for Normalization
        nodes = np.array([[d['x'], d['y']] for _, d in self.G.nodes(data=True)])
        self.max_dist = np.linalg.norm(nodes.max(axis=0) - nodes.min(axis=0))

        # 3. STATE SPACE (The "Eyes")
        # [Dist_to_Goal, Deviation_Road_0, Deviation_Road_1, Deviation_Road_2, Loop_Warning]
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        
        # 4. ACTION SPACE (Relative Choices)
        # 0: Best Road, 1: 2nd Best Road, 2: 3rd Best Road
        self.action_space = gym.spaces.Discrete(3) 
        
        self.current_node = self.start_node
        self.visited = set()

    def get_state(self):
        curr_data = self.G.nodes[self.current_node]
        tgt_data = self.G.nodes[self.target_node]
        
        # A. Distance Feature (0.0 to 1.0)
        dist = self.get_distance(self.current_node, self.target_node) / self.max_dist
        
        # B. Heading Feature
        # Calculate the "Ideal Bearing" to the target
        bearing_to_goal = ox.bearing.calculate_bearing(
            curr_data['y'], curr_data['x'], tgt_data['y'], tgt_data['x']
        )
        
        # C. Road Features (The "Smart" Sorter)
        # Get all neighbors and sort them by how close they align to the goal
        neighbors = list(self.G.successors(self.current_node))
        road_deviations = []
        
        neighbor_props = []
        for n in neighbors:
            n_data = self.G.nodes[n]
            road_bearing = ox.bearing.calculate_bearing(
                curr_data['y'], curr_data['x'], n_data['y'], n_data['x']
            )
            # Calculate deviation (0 = Perfect alignment, 180 = Opposite direction)
            diff = abs(road_bearing - bearing_to_goal)
            diff = min(diff, 360 - diff)
            neighbor_props.append((diff, n))
            
        # Sort neighbors: Best aligned first
        neighbor_props.sort(key=lambda x: x[0])
        self.sorted_neighbors = [n[1] for n in neighbor_props]
        
        # Fill State Vector (Top 3 roads)
        for i in range(3):
            if i < len(neighbor_props):
                # Normalize deviation (0.0 to 1.0) where 0 is good
                road_deviations.append(neighbor_props[i][0] / 180.0)
            else:
                road_deviations.append(1.0) # No road = Bad deviation
        
        # D. Loop Warning
        # Is the 'Best Road' one we just visited? (1.0 = Yes, 0.0 = No)
        loop_warning = 0.0
        if len(self.sorted_neighbors) > 0 and self.sorted_neighbors[0] in self.visited:
            loop_warning = 1.0

        state = np.array([dist] + road_deviations + [loop_warning], dtype=np.float32)
        return state

    def step(self, action):
        reward = -0.1 # Step Cost (Time)
        terminated = False
        
        # Execute Action (Pick from sorted list)
        if action < len(self.sorted_neighbors):
            next_node = self.sorted_neighbors[action]
            
            # --- REWARD SHAPING ---
            
            # 1. Loop Penalty (Crucial for not getting stuck)
            if next_node in self.visited:
                reward -= 0.5
            else:
                self.visited.add(next_node)
            
            # 2. Progress Reward (Euclidean)
            prev_dist = self.get_distance(self.current_node, self.target_node)
            curr_dist = self.get_distance(next_node, self.target_node)
            # Reward moving closer, penalize moving away
            reward += (prev_dist - curr_dist) * 20 
            
            # Move
            self.current_node = next_node
            
            if self.current_node == self.target_node:
                reward += 50 # Goal Reached
                terminated = True
        else:
            # Tried to pick a road that doesn't exist (e.g., Action 2 on a 2-way street)
            reward -= 1.0 
        
        truncated = False
        return self.get_state(), reward, terminated, truncated, {}

    def get_distance(self, u, v):
        n = self.G.nodes[u]
        t = self.G.nodes[v]
        return np.sqrt((n['x']-t['x'])**2 + (n['y']-t['y'])**2)

    def reset(self):
        self.current_node = self.start_node
        self.visited = set()
        self.visited.add(self.current_node)
        return self.get_state()

# --- 2. THE BRAIN (Actor-Critic) ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Input Layer
        self.fc1 = nn.Linear(state_dim, 64)
        
        # Actor Head (Outputs Probabilities)
        self.actor = nn.Linear(64, action_dim)
        
        # Critic Head (Outputs Value Estimate)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)

# --- 3. TRAINING LOOP ---
def train():
    env = AgriFlowEnv()
    agent = ActorCritic(state_dim=5, action_dim=3).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    
    episodes = 500
    print("Training on Micro-Map...")
    
    for episode in tqdm(range(episodes)):
        state = env.reset()
        state = torch.FloatTensor(state).to(DEVICE)
        
        log_probs = []
        values = []
        rewards = []
        done = False
        steps = 0
        
        while not done and steps < 100:
            action_probs, value = agent(state)
            
            # Sample Action
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            next_state, reward, done, _, _ = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            
            state = torch.FloatTensor(next_state).to(DEVICE)
            steps += 1
            
        # Backpropagation (A2C Algorithm)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(DEVICE)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = 0
        value_loss = 0
        for log_prob, value, R in zip(log_probs, values, returns):
            advantage = R - value.item()
            policy_loss -= log_prob * advantage
            value_loss += F.smooth_l1_loss(value, torch.tensor([R]).to(DEVICE).unsqueeze(0))
            
        optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        optimizer.step()
        
    return agent, env

# --- 4. VISUALIZATION ---
def visualize(agent, env):
    state_np = env.reset()
    state = torch.FloatTensor(state_np).to(DEVICE)
    path = [env.current_node]
    done = False
    
    print("Generating Path...")
    steps = 0
    while not done and steps < 100:
        with torch.no_grad():
            action_probs, _ = agent(state)
            # Use Argmax for final test (Best Action)
            action = torch.argmax(action_probs).item()
            
        state_np, _, done, _, _ = env.step(action)
        state = torch.FloatTensor(state_np).to(DEVICE)
        
        if env.current_node != path[-1]:
            path.append(env.current_node)
        steps += 1
        
    print(f"Path Length: {len(path)}")
    
    # Plotting
    if len(path) > 1:
        # Use ox.plot_graph_route for clean visualization
        fig, ax = ox.plot_graph_route(env.G, path, route_linewidth=4, node_size=30, bgcolor='white', edge_color='gray', node_color='blue', show=False, close=False)
        plt.savefig(os.path.join(VISUALIZATIONS_DIR, "agriflow_micro.png"))
        print("Map saved to 'agriflow_micro.png'. Open this file to see the pixels!")
    else:
        print("Agent stuck.")

if __name__ == "__main__":
    trained_agent, environment = train()
    visualize(trained_agent, environment)