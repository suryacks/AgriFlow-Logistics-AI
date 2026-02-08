
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loader import ScenarioLoader
from environment import AgriFlowFleetEnv
import numpy as np
import os
import gymnasium as gym

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'resources', 'agriflow_fleet_agent.pth')

class FleetPolicy(nn.Module):
    def __init__(self, input_dim, n_sources, n_dests):
        super(FleetPolicy, self).__init__()
        # Shared Trunk
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Head 1: Select Source
        self.head_src = nn.Linear(128, n_sources)
        
        # Head 2: Select Destination
        self.head_dst = nn.Linear(128, n_dests)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(x))
        
        # Logic: We return Logits for both heads
        logits_src = self.head_src(feat)
        logits_dst = self.head_dst(feat)
        
        return F.softmax(logits_src, dim=-1), F.softmax(logits_dst, dim=-1)

def train_fleet():
    # 1. Load Data
    loader = ScenarioLoader()
    graph, suggested_fleet_size = loader.load_scenario()
    
    # 2. Env (Use Suggested Fleet Size! Likely ~30-50 trucks)
    env = AgriFlowFleetEnv(graph, num_trucks=suggested_fleet_size)
    
    # 3. Agent
    input_dim = env.observation_space.shape[0]
    # action_space is MultiDiscrete([n_supply, n_demand])
    n_supply = env.action_space.nvec[0]
    n_demand = env.action_space.nvec[1]
    
    agent = FleetPolicy(input_dim, n_supply, n_demand)
    optimizer = optim.Adam(agent.parameters(), lr=0.0005)
    
    print(f"--- AGRIFLOW FLEET TRAINING ---")
    print(f"Observation Dim: {input_dim}")
    print(f"Action Heads: Source({n_supply}), Dest({n_demand})")
    
    # 4. Loop
    episodes = 500
    for episode in range(episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False
        steps = 0
        
        while not done and steps < 100:
            state_t = torch.FloatTensor(state)
            
        while not done and steps < 100:
            state_t = torch.FloatTensor(state)
            
            # Forward Pass (Get both logits)
            prob_src, prob_dst = agent(state_t)
            
            # --- STEP 1: SELECT SOURCE ---
            src_mask = env.get_source_mask()
            prob_src = prob_src * src_mask
            if prob_src.sum() > 0: prob_src /= prob_src.sum()
            else: prob_src = torch.ones_like(prob_src) / len(prob_src)
            
            dist_src = torch.distributions.Categorical(prob_src)
            a_src = dist_src.sample()
            
            # --- STEP 2: SELECT DESTINATION (Refined by Source) ---
            dst_mask = env.get_dest_mask(a_src.item())
            prob_dst = prob_dst * dst_mask
            if prob_dst.sum() > 0: prob_dst /= prob_dst.sum()
            else: prob_dst = torch.ones_like(prob_dst) / len(prob_dst)
            
            dist_dst = torch.distributions.Categorical(prob_dst)
            a_dst = dist_dst.sample()
            
            # Action Tuple
            action = np.array([a_src.item(), a_dst.item()])
            
            next_state, reward, done, _, _ = env.step(action)
            
            # Store Log Probs
            log_prob = dist_src.log_prob(a_src) + dist_dst.log_prob(a_dst)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            state = next_state
            steps += 1
            
        # PPO/REINFORCE Update
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + 0.95 * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        loss_val = 0
        for log_prob, R in zip(log_probs, returns):
            loss_val -= log_prob * R
            
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
        if episode % 50 == 0:
            profit = sum(rewards)
            print(f"Episode {episode} | Steps {steps} | Fleet Profit: ${profit:.2f}")

    torch.save(agent.state_dict(), MODEL_PATH)
    print("Fleet Agent Saved.")

if __name__ == "__main__":
    train_fleet()
