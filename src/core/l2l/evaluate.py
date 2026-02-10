import torch
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import *
from environment import AgriFlowEnv
from cortex import AgriFlowBrain

def calculate_path_time(G, path):
    """Helper to calculate total travel time of a path manually."""
    total_time = 0
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data:
            # MultiDiGraph returns a dict of edges {key: attributes}
            # We take the first available edge (usually key=0)
            edge = edge_data[min(edge_data.keys())]
            total_time += edge.get('travel_time', 0)
    return total_time

def evaluate_and_compare():
    env = AgriFlowEnv()
    
    # Load the Trained Brain
    agent = AgriFlowBrain(hidden_size=HIDDEN_SIZE).to(DEVICE)
    try:
        agent.load_state_dict(torch.load(MODEL_FILENAME))
        print("AgriFlow Cortex Loaded Successfully.")
    except FileNotFoundError:
        print("Error: Model file not found. Run train.py first!")
        return

    agent.eval() # Set to evaluation mode
    
    print(f"\n--- STARTING GROUND TRUTH COMPARISON ({TEST_ROUNDS} Rounds) ---")
    
    score_card = []

    for i in range(TEST_ROUNDS):
        env.set_random_mission()
        print(f"\nTest {i+1}: Node {env.start_node} -> Node {env.target_node}")
        
        # 1. GENERATE GOOGLE MAPS (Ground Truth) PATH
        try:
            gt_path = nx.shortest_path(env.G, env.start_node, env.target_node, weight='travel_time')
            gt_len = len(gt_path)
            gt_time = calculate_path_time(env.G, gt_path)
        except nx.NetworkXNoPath:
            print("Skipping: No path possible.")
            continue

        # 2. GENERATE AGRIFLOW PATH
        state = env.reset()
        state = torch.FloatTensor(state).to(DEVICE)
        ag_path = [env.current_node]
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            with torch.no_grad():
                action_probs, _ = agent(state)
                action = torch.argmax(action_probs).item() # Greedy (Best) Action
            
            next_state, _, done, _, _ = env.step(action)
            state = torch.FloatTensor(next_state).to(DEVICE)
            
            if env.current_node != ag_path[-1]:
                ag_path.append(env.current_node)
            steps += 1
        
        # 3. COMPARE
        ag_len = len(ag_path)
        ag_time = calculate_path_time(env.G, ag_path)
            
        success = (ag_path[-1] == env.target_node)
        
        # Efficiency Score: (Ground Truth Time / AgriFlow Time) * 100
        # If ag_time is 0 (didn't move), efficiency is 0.
        efficiency = (gt_time / ag_time) * 100 if ag_time > 0 else 0
        
        print(f"   [Google Maps] Length: {gt_len} | Time: {gt_time:.2f}s")
        print(f"   [AgriFlow]    Length: {ag_len} | Time: {ag_time:.2f}s | Success: {success}")
        print(f"   >>> Efficiency Rating: {efficiency:.1f}%")
        
        score_card.append(efficiency)

        # 4. VISUALIZE COMPARISON (Save the last one)
        if i == TEST_ROUNDS - 1:
            try:
                fig, ax = ox.plot_graph_routes(env.G, [gt_path, ag_path], 
                                             route_colors=['blue', 'green'], 
                                             route_linewidths=[6, 4], 
                                             node_size=0, bgcolor='black', show=False, close=False)
                plt.savefig(os.path.join(VISUALIZATIONS_DIR, "comparison_result.png"))
                print("\nSaved comparison map to 'comparison_result.png' (Blue=Google, Green=AgriFlow)")
            except Exception as e:
                print(f"Visualization failed: {e}")

    print(f"\n--- FINAL REPORT ---")
    if score_card:
        print(f"Average Efficiency across {len(score_card)} routes: {sum(score_card)/len(score_card):.1f}%")
    else:
        print("No valid routes tested.")

if __name__ == "__main__":
    evaluate_and_compare()