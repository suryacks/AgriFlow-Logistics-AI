import torch
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import random
import os

from config import *
from environment import AgriFlowEnv
from cortex import AgriFlowBrain

def run_single_visual_test():
    print("--- INITIATING MANUAL AGRIFLOW TEST ---")
    env = AgriFlowEnv()
    
    # 1. Load the Brain
    agent = AgriFlowBrain(hidden_size=HIDDEN_SIZE).to(DEVICE)
    if os.path.exists(MODEL_FILENAME):
        agent.load_state_dict(torch.load(MODEL_FILENAME, map_location=DEVICE))
        print("AgriFlow Cortex Loaded Successfully.")
    else:
        print(f"CRITICAL ERROR: No model found at {MODEL_FILENAME}. Run 'train.py' first!")
        return

    agent.eval()
    
    # 2. Setup Random Mission
    # We force a random mission to prove the AI isn't memorizing.
    env.set_random_mission()
    print(f"Mission: Node {env.start_node} -> Node {env.target_node}")

    # 3. Get Ground Truth (Google Maps Proxy - Blue)
    print("Calculating Ground Truth (Dijkstra)...")
    try:
        gt_path = nx.shortest_path(env.G, env.start_node, env.target_node, weight='travel_time')
    except nx.NetworkXNoPath:
        print("Error: No physical path exists between these points. Try again.")
        return

    # 4. Get AgriFlow Path (AI - Red)
    print("AgriFlow AI Navigating...")
    state = env.reset()
    state = torch.FloatTensor(state).to(DEVICE)
    ai_path = [env.current_node]
    done = False
    steps = 0
    
    while not done and steps < MAX_STEPS * 2:
        with torch.no_grad():
            action_probs, _ = agent(state)
            action = torch.argmax(action_probs).item()
        
        next_state, _, done, _, _ = env.step(action)
        state = torch.FloatTensor(next_state).to(DEVICE)
        
        # Track path only if we moved
        if env.current_node != ai_path[-1]:
            ai_path.append(env.current_node)
        steps += 1
        
    # 5. Success Check
    success = (ai_path[-1] == env.target_node)
    print(f"Result: {'SUCCESS' if success else 'FAILURE'}")
    print(f"AI Path Length: {len(ai_path)} | Ground Truth Length: {len(gt_path)}")

    # 6. Visualization (Blue vs Red)
    print("Generating Comparison Map...")
    filename = os.path.join(VISUALIZATIONS_DIR, "manual_test_result.png")
    
    # Plot routes with distinct colors and widths
    # Ground Truth = Blue (Thicker)
    # AI = Red (Thinner, on top)
    fig, ax = ox.plot_graph_routes(
        env.G, 
        [gt_path, ai_path], 
        route_colors=['blue', 'red'],      
        route_linewidths=[6, 3],           
        node_size=0, 
        bgcolor='white', 
        edge_color='#999999',
        show=False, 
        close=False
    )
    
    # Add Start/End Markers for clarity
    start_xy = (env.G.nodes[env.start_node]['x'], env.G.nodes[env.start_node]['y'])
    end_xy = (env.G.nodes[env.target_node]['x'], env.G.nodes[env.target_node]['y'])
    
    ax.scatter(start_xy[0], start_xy[1], c='green', s=150, zorder=10, label='Start')
    ax.scatter(end_xy[0], end_xy[1], c='black', s=150, marker='X', zorder=10, label='End')
    
    plt.legend()
    plt.savefig(filename)
    print(f"Test Complete. Saved map to '{filename}'")
    plt.close()

if __name__ == "__main__":
    run_single_visual_test()