
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loader import ScenarioLoader
from environment import AgriFlowFleetEnv
import numpy as np
import os
import random

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'resources', 'agriflow_fleet_agent.pth')

class FleetPolicy(nn.Module):
    def __init__(self, input_dim, n_sources, n_dests):
        super(FleetPolicy, self).__init__()
        
        # Attention Mechanism for Features?
        # For now, simple deep network with Dropout to handle noise (Traffic)
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 256)
        
        self.head_src = nn.Linear(256, n_sources)
        self.head_dst = nn.Linear(256, n_dests)
        
    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        feat = F.elu(self.fc2(x))
        # Return Logits!
        return self.head_src(feat), self.head_dst(feat)

def compare_fleet_performance():
    # 1. Load Simulation
    loader = ScenarioLoader()
    graph, fleet_size = loader.load_scenario()
    
    # DISABLE TRAFFIC for "Normal Day" Comparison
    env = AgriFlowFleetEnv(graph, num_trucks=fleet_size, enable_traffic=False)
    
    print(f"\n--- FLEET SHOWDOWN (NORMAL DAY): {fleet_size} TRUCKS ---")
    
    # 2. Run TRADITIONAL (Greedy + Static)
    print("[Traditional] Running Logistics...")
    state, _ = env.reset()
    sap_profit = 0
    done = False
    
    while not done:
        # Greedy Logic:
        # Pick FIRST truck. Find Best Valid Route.
        mask_src = env.get_source_mask()
        
        src_indices = torch.nonzero(mask_src).flatten().tolist()
        best_action = (0, 0)
        max_score = -float('inf')
        
        if len(src_indices) > 0:
            # Optimize: Check top 10 sources max
            check_src = random.sample(src_indices, min(10, len(src_indices)))
            for src_idx in check_src:
                mask_dst_local = env.get_dest_mask(src_idx)
                dst_indices = torch.nonzero(mask_dst_local).flatten().tolist()
                
                if len(dst_indices) > 0:
                    for dst_idx in dst_indices:
                         s_id = env.supply_nodes[src_idx]
                         d_id = env.demand_nodes[dst_idx]
                         
                         if not env.graph.has_edge(s_id, d_id): continue
                         
                         edge = env.graph[s_id][d_id]
                         dist = edge.get('distance', 50.0)
                         
                         # Greedy Score: Revenue - Cost
                         score = (10000) - (dist * 1.5)
                         if score > max_score:
                             max_score = score
                             best_action = (src_idx, dst_idx)
            else:
                pass
        
        if best_action == (0,0) and len(src_indices) > 0:
             # Fallback
             best_action = (src_indices[0], 0)
             
        # Execute
        _, r, done, _, _ = env.step(best_action)
        sap_profit += r
        
    print(f"Traditional Profit: ${sap_profit:,.2f}")
    
    # 3. Run AGRIFLOW (RL Agent)
    print("[AgriFlow] Training Optimization Swarm (50 Eps)...")
    input_dim = env.observation_space.shape[0]
    n_s = env.action_space.nvec[0]
    n_d = env.action_space.nvec[1]
    agent = FleetPolicy(input_dim, n_s, n_d)
    
    # Train harder
    optimizer = optim.Adam(agent.parameters(), lr=0.002)
    
    for ep in range(50):
        s, _ = env.reset()
        done = False
        steps = 0
        total_r = 0
        while not done and steps < 200:
            st = torch.FloatTensor(s)
            
            # 1. Get Logits
            logits_s, logits_d = agent(st)
            
            # 2. Mask Source Logits
            ms = env.get_source_mask()
            # Set invalid items to -inf
            logits_s = logits_s.masked_fill(ms == 0, float('-1e9'))
            
            p_s = F.softmax(logits_s, dim=-1)
            dist_src = torch.distributions.Categorical(p_s)
            asrc = dist_src.sample()
            
            # 3. Mask Dest Logits
            md = env.get_dest_mask(asrc.item())
            logits_d = logits_d.masked_fill(md == 0, float('-1e9'))
            
            p_d = F.softmax(logits_d, dim=-1)
            dist_dst = torch.distributions.Categorical(p_d)
            adst = dist_dst.sample()
            
            # Step
            ns, r, done, _, _ = env.step((asrc.item(), adst.item()))
            total_r += r
            
            # Loss
            log_prob = dist_src.log_prob(asrc) + dist_dst.log_prob(adst)
            loss = -log_prob * r
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            s = ns
            steps += 1
        if ep % 10 == 0: print(f"Ep {ep}: ${total_r:,.0f}")
            
    # Evaluation Run
    print("[AgriFlow] Engaging Swarm...")
    s, _ = env.reset()
    ai_profit = 0
    done = False
    with torch.no_grad():
        while not done:
            st = torch.FloatTensor(s)
            logits_s, logits_d = agent(st)
            
            # Masking
            ms = env.get_source_mask()
            logits_s = logits_s.masked_fill(ms == 0, float('-1e9'))
            p_s = F.softmax(logits_s, dim=-1)
            
            asrc = torch.argmax(p_s) # Greedy AI
            
            md = env.get_dest_mask(asrc.item())
            logits_d = logits_d.masked_fill(md == 0, float('-1e9'))
            p_d = F.softmax(logits_d, dim=-1)
            
            adst = torch.argmax(p_d)
            
            _, r, done, _, _ = env.step((asrc.item(), adst.item()))
            ai_profit += r
            
    print(f"AgriFlow Profit:   ${ai_profit:,.2f}")
    
    if ai_profit > sap_profit:
        print("\n>>> VICTORY: AgriFlow destroyed Traditional Logistics strategies.")
    else:
        print(f"\n>>> DEFEAT: AgriFlow {ai_profit} vs SAP {sap_profit}")

if __name__ == "__main__":
    compare_fleet_performance()
