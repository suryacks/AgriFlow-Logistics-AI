
import sys
import os

# Ensure we can import from the directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the environment (which loads the map)
# We assume the user has the map file or we generate it.
# The legacy code had 'GRAPH_FILENAME' in config.py
# We might need to handle the config import.

try:
    from .environment import AgriFlowEnv
    ENV_AVAILABLE = True
except ImportError:
    print("Warning: L2L Dependencies missing (gym, osmnx). Running in Mock Mode.")
    ENV_AVAILABLE = False

class L2LInterface:
    """
    Wrapper for the Legacy Lane-to-Lane (L2L) Reinforcement Learning Engine.
    Provides route complexity scores.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(L2LInterface, cls).__new__(cls)
            cls._instance.env = None
            if ENV_AVAILABLE:
                try:
                    # Initialize Env (Heavy Loading)
                    # We defer this to first call or handle carefully
                    pass 
                except Exception as e:
                    print(f"L2L Init Failed: {e}")
        return cls._instance

    def get_route_complexity(self, start_loc, end_loc):
        """
        Returns a complexity score (0.0 - 1.0).
        High complexity = difficult route (potential alpha source).
        """
        # For the demo, we simulate the complexity based on length
        # In full production, this would run the RL Agent.
        
        # Mock Logic for Speed
        import random
        base_score = random.uniform(0.3, 0.7)
        
        if "Snow" in start_loc or "Ice" in start_loc: # Context-aware mocking
            base_score += 0.2
            
        return min(base_score, 1.0)

    def run_grid_demo(self, nodes_count=40, obstacles=[]):
        """
        Generates a "City Map" (Random Geometric Graph) and finds path.
        Simulates the L2L Agent navigating a road network.
        """
        import networkx as nx
        import random
        
        # 1. Generate Graph
        # Use seed so the map looks the same every time until refreshed
        random.seed(2024) 
        G = nx.random_geometric_graph(nodes_count, 0.25)
        
        # Ensure connectivity (take largest component)
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(components[0]).copy()
        
        # Scale positions: 0.0-1.0 -> 0-100 logic (Frontend handles canvas scaling)
        pos = nx.get_node_attributes(G, 'pos')
        
        # Convert to list
        nodes_data = []
        node_map = {} # Map original ID to new index if needed? 
        # Actually random_geometric_graph uses int labels 0..N
        
        sorted_nodes = sorted(list(G.nodes))
        
        for n in sorted_nodes:
            nodes_data.append({
                "id": n,
                "x": pos[n][0],
                "y": pos[n][1]
            })
            
        edges_data = [{"u": u, "v": v} for u,v in G.edges]
        
        # 2. Heuristic Start/End
        # Pick Left-most and Right-most
        sorted_by_x = sorted(nodes_data, key=lambda n: n['x'])
        start_node = sorted_by_x[0]['id']
        end_node = sorted_by_x[-1]['id']
        
        # 3. Apply Obstacles (Node Removal)
        # Obstacles are passed as Node IDs from frontend
        G_path = G.copy()
        active_obstacles = []
        
        for o in obstacles:
            # o is node ID
            if o in G_path:
                G_path.remove_node(o)
                active_obstacles.append(o)
                
        # 4. Pathfinding
        try:
            path = nx.shortest_path(G_path, source=start_node, target=end_node)
        except:
            path = []
            
        return {
            "type": "map",
            "nodes": nodes_data,
            "edges": edges_data,
            "path": path,
            "start": start_node,
            "end": end_node,
            "obstacles": active_obstacles
        }
