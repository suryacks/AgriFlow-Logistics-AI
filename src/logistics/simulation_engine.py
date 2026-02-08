
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from .loader import ScenarioLoader
from .environment import AgriFlowFleetEnv

class FleetPolicy(nn.Module):
    def __init__(self, input_dim, n_sources, n_dests):
        super(FleetPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(512, 256)
        self.head_src = nn.Linear(256, n_sources)
        self.head_dst = nn.Linear(256, n_dests)
        
    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        feat = F.elu(self.fc2(x))
        return self.head_src(feat), self.head_dst(feat)

class SimulationEngine:
    def __init__(self):
        self.loader = ScenarioLoader()
        self.graph, self.fleet_size = self.loader.load_scenario()

    def run_simulation(self, fleet_size_input=None, enable_traffic=True, heat_start=0.5, traffic_prob=0.1, simulation_steps=500):
        f_size = fleet_size_input if fleet_size_input else self.fleet_size
        
        # Setup Environments
        env = AgriFlowFleetEnv(self.graph, num_trucks=f_size, enable_traffic=enable_traffic)
        
        # Traditional Run (Baseline)
        state, _ = env.reset(heat_start=heat_start, traffic_prob=traffic_prob)
        sap_cum_profit = []
        current_sap_profit = 0
        env.step_count = 0
        done = False
        step = 0
        
        while not done and step < simulation_steps:
             mask_src = env.get_source_mask()
             src_indices = torch.nonzero(mask_src).flatten().tolist()
             best_action = (0, 0)
             max_score = -float('inf')
             
             if len(src_indices) > 0:
                 check_src = random.sample(src_indices, min(5, len(src_indices)))
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
                             score = (10000) - (dist * 1.5)
                             if score > max_score:
                                 max_score = score
                                 best_action = (src_idx, dst_idx)
             
             if best_action == (0,0) and len(src_indices) > 0:
                  best_action = (src_indices[0], 0)
                  
             _, r, done, _, _ = env.step(best_action)
             current_sap_profit += r
             sap_cum_profit.append(current_sap_profit)
             step += 1
             
        # AgriFlow Run (RL Agent)
        input_dim = env.observation_space.shape[0]
        n_s = env.action_space.nvec[0]
        n_d = env.action_space.nvec[1]
        agent = FleetPolicy(input_dim, n_s, n_d)
        optimizer = optim.Adam(agent.parameters(), lr=0.002)
        
        # Warm-up Phase
        for _ in range(5):
             s, _ = env.reset(heat_start=heat_start, traffic_prob=traffic_prob)
             d = False
             stp = 0
             while not d and stp < 50:
                 st = torch.FloatTensor(s)
                 l_s, l_d = agent(st)
                 
                 ms = env.get_source_mask()
                 l_s = l_s.masked_fill(ms == 0, float('-1e9'))
                 p_s = F.softmax(l_s, dim=-1)
                 asrc = torch.distributions.Categorical(p_s).sample()
                 
                 md = env.get_dest_mask(asrc.item())
                 l_d = l_d.masked_fill(md == 0, float('-1e9'))
                 p_d = F.softmax(l_d, dim=-1)
                 adst = torch.distributions.Categorical(p_d).sample()
                 
                 ns, r, d, _, _ = env.step((asrc.item(), adst.item()))
                 
                 if l_s.dim() == 2:
                     log_p_s = torch.log(p_s[0, asrc])
                     log_p_d = torch.log(p_d[0, adst])
                 else:
                     log_p_s = torch.log(p_s[asrc])
                     log_p_d = torch.log(p_d[adst])
                     
                 loss = -(log_p_s + log_p_d) * r
                 
                 optimizer.zero_grad()
                 loss.backward()
                 optimizer.step()
                 s = ns
                 stp += 1
        
        # Evaluation Phase
        state, _ = env.reset(heat_start=heat_start, traffic_prob=traffic_prob)
        agri_cum_profit = []
        current_agri_profit = 0
        done = False
        step = 0
        
        while not done and step < simulation_steps:
            st = torch.FloatTensor(state)
            with torch.no_grad():
                l_s, l_d = agent(st)
                
                ms = env.get_source_mask()
                l_s = l_s.masked_fill(ms == 0, float('-1e9'))
                p_s = F.softmax(l_s, dim=-1)
                asrc = torch.argmax(p_s)
                
                md = env.get_dest_mask(asrc.item())
                l_d = l_d.masked_fill(md == 0, float('-1e9'))
                p_d = F.softmax(l_d, dim=-1)
                adst = torch.argmax(p_d)
                
            _, r, done, _, _ = env.step((asrc.item(), adst.item()))
            current_agri_profit += r
            agri_cum_profit.append(current_agri_profit)
            step += 1
            
        min_len = min(len(sap_cum_profit), len(agri_cum_profit))
        df = pd.DataFrame({
            'Step': range(min_len),
            'Traditional (SAP)': sap_cum_profit[:min_len],
            'AgriFlow (AI)': agri_cum_profit[:min_len]
        })
        
    def obtain_stress_score(self, traffic_intensity, heat_intensity):
        """
        Runs the AgileFlow RL Agent on a 'Digital Twin' of the specified weather conditions.
        Returns a 'Logistics Stress Score' (0-100).
        
        0 = Smooth Sailing
        100 = Total Supply Chain Collapse
        """
        # 1. Run Baseline (Perfect Conditions)
        _, base_sap, base_ai = self.run_simulation(fleet_size_input=50, enable_traffic=False, heat_start=0.0, traffic_prob=0.0, simulation_steps=100)
        
        # 2. Run Stress Scenario
        # Map inputs: traffic_intensity (0-1) -> traffic_prob (0.0-0.5)
        # heat_intensity (0-1) -> heat_start (0.0-1.0)
        t_prob = traffic_intensity * 0.5
        h_start = heat_intensity
        
        _, stress_sap, stress_ai = self.run_simulation(fleet_size_input=50, enable_traffic=True, heat_start=h_start, traffic_prob=t_prob, simulation_steps=100)
        
        # 3. Calculate "Disruption"
        # We compare how much the Traditional System FAILED.
        # This is the "Opportunity for Arbitrage".
        # If Traditional Profit crashes, prices will spike.
        
        # Protect div/0
        if base_sap == 0: base_sap = 1
        
        drop = (base_sap - stress_sap) / abs(base_sap)
        
        # Normalize to 0-100 score
        # A drop of 150% (negative profit) = 100 score
        # A drop of 0% = 0 score
        score = min(max(drop * 66, 0), 100)
        
        return score, stress_ai, stress_sap
