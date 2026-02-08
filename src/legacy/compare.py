
import torch
import numpy as np
from loader import ScenarioLoader
from environment import LogisticsEnv
from train_logistics import PolicyNet, MODEL_PATH
import os

def run_sap_greedy_simulation(env):
    """
    Simulates a traditional 'Greedy' optimizer (SAP-style approximate).
    Logic: Always pick the cheapest valid transportation lane that moves milk.
    """
    state = env.reset()
    done = False
    total_profit = 0
    steps = 0
    
    print("\n[Traditional Optimizer] Calculating Schedule...")
    
    current_supply_map = env.supply_nodes 
    # Note: Environment rotates supply nodes automatically. We must follow its lead.
    
    while not done and steps < 200:
        # 1. Identify Valid Options
        current_supply_id = env.supply_nodes[env.active_supply_idx]
        mask = env.get_valid_actions_mask()
        
        best_action = -1
        best_value = -float('inf')
        
        # Brute force search for "Cheapest Cost" among valid neighbors
        neighbors = list(env.graph.successors(current_supply_id))
        
        # Map neighbor IDs to action indices
        # This is slow but mimics the solver checking all routes
        valid_indices = torch.nonzero(mask).flatten().tolist()
        
        if not valid_indices:
            # No moves
            best_action = 0
        else:
            # Pick the neighbor with lowest cost that NEEDS milk
            # Solver Heuristic: Minimize Cost, Maximize fill
            best_score = -float('inf')
            
            for idx in valid_indices:
                target_id = env.demand_nodes[idx]
                
                # Safety check (Mask might be all 1s if fallback triggered)
                if not env.graph.has_edge(current_supply_id, target_id):
                    continue
                    
                edge = env.graph[current_supply_id][target_id]
                cost = edge['cost']
                
                needed = env.current_demand[target_id]
                
                if needed > 0:
                    # CALCULATE REALISTIC PROFIT
                    dist = edge.get('distance', 50.0)
                    dur = edge.get('duration', 1.0)
                    fuel_cost = dist * 0.70
                    labor_cost = dur * 30.0
                    transport_cost = fuel_cost + labor_cost
                    
                    revenue = 20000 * 0.05 # Assume full truck service value
                    score = revenue - transport_cost
                else:
                    score = -1000 # Do not move empty in greedy simulation
                
                if score > best_score:
                    best_score = score
                    best_action = idx
            
            if best_action == -1:
                best_action = valid_indices[0] # Fallback
                
        # Execute
        _, reward, done, _, _ = env.step(best_action)
        total_profit += reward
        steps += 1
        
    return total_profit, steps

def run_agriflow_ai(env):
    """
    Runs the Trained RL Agent.
    """
    # Load Model
    input_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = PolicyNet(input_dim, n_actions)
    
    try:
        agent.load_state_dict(torch.load(MODEL_PATH))
        agent.eval()
    except:
        print("Error: Could not load trained model. Train first!")
        return 0, 0

    state = env.reset()
    done = False
    total_profit = 0
    steps = 0
    
    print("\n[AgriFlow AI] Optimizing Distribution...")
    
    while not done and steps < 200:
        state_t = torch.FloatTensor(state)
        mask = env.get_valid_actions_mask()
        
        with torch.no_grad():
            action_probs = agent(state_t, mask)
            action = torch.argmax(action_probs).item() # GREEDY choice from AI
        
        _, reward, done, _, _ = env.step(action)
        total_profit += reward
        steps += 1
        
    return total_profit, steps

def compare_systems():
    loader = ScenarioLoader()
    graph = loader.load_scenario()
    env = LogisticsEnv(graph)
    
    # Run Baseline
    sap_profit, sap_steps = run_sap_greedy_simulation(env)
    
    # Run AI
    ai_profit, ai_steps = run_agriflow_ai(env)
    
    print("\n" + "="*40)
    print("      LOGISTICS SYSTEM SHOWDOWN      ")
    print("="*40)
    print(f"TRADITIONAL (Rule-Based):")
    print(f"   Profit: ${sap_profit:,.2f}")
    print(f"   Steps:  {sap_steps}")
    print("-" * 40)
    print(f"AGRIFLOW (Reinforcement Learning):")
    print(f"   Profit: ${ai_profit:,.2f}")
    print(f"   Steps:  {ai_steps}")
    print("-" * 40)
    
    diff = ai_profit - sap_profit
    if diff > 0:
        print(f"RESULT: AgriFlow generated ${diff:,.2f} MORE profit.")
    else:
        print(f"RESULT: AgriFlow is lagging by ${abs(diff):,.2f}. Needs more training.")
    print("="*40)

if __name__ == "__main__":
    compare_systems()
