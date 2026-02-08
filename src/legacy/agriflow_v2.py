
import random
import numpy as np
import networkx as nx
from loader import ScenarioLoader
import time

class AgriFlowV2:
    def __init__(self):
        print("Initialization AgriFlow V2: The Advanced Logistics Engine...")
        self.loader = ScenarioLoader()
        self.graph, self.fleet_size = self.loader.load_scenario()
        
        self.supply_nodes = sorted([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'supply'])
        self.demand_nodes = sorted([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'demand'])
        
        print(f"Network: {len(self.supply_nodes)} Farms, {len(self.demand_nodes)} Customers, {self.fleet_size} Trucks.")
        
    def calculate_cost(self, individual):
        """
        Evaluate a Plan (Individual).
        Plan = List of (Source, Dest) pairs for each truck.
        """
        total_profit = 0
        
        # Snapshot of inventory
        current_supply = {n: self.graph.nodes[n]['data'].available_qty for n in self.supply_nodes}
        current_demand = {n: self.graph.nodes[n]['data'].daily_capacity for n in self.demand_nodes}
        
        for (src_idx, dst_idx) in individual:
            src_id = self.supply_nodes[src_idx]
            dst_id = self.demand_nodes[dst_idx]
            
            if self.graph.has_edge(src_id, dst_id):
                edge = self.graph[src_id][dst_id]
                dist = edge.get('distance', 50.0)
                
                # Logic
                avail = current_supply.get(src_id, 0)
                need = current_demand.get(dst_id, 0)
                
                moved = min(avail, need, 20000.0)
                
                if moved > 0:
                    revenue = moved * 0.50
                    cost = dist * 1.50
                    profit = revenue - cost
                    
                    total_profit += profit
                    
                    # Update state
                    current_supply[src_id] -= moved
                    current_demand[dst_id] -= moved
                else:
                    total_profit -= 50 # Empty trip penalty
            else:
                total_profit -= 500 # Invalid Route
                
        return total_profit

    def run_genetic_algorithm(self, generations=100, population_size=50):
        print(f"--- STARTING GENETIC OPTIMIZER (Simulating SAP) ---")
        
        # 1. Initialize Population (Random Valid Plans)
        population = []
        for _ in range(population_size):
            plan = []
            for _ in range(self.fleet_size):
                # Try to pick a VALID random pair to start
                s = random.randint(0, len(self.supply_nodes)-1)
                # Pick a connected neighbor if possible?
                # For GA, we can start random and evolve.
                d = random.randint(0, len(self.demand_nodes)-1)
                plan.append((s, d))
            population.append(plan)
            
        # 2. Evolution Loop
        for gen in range(generations):
            # Evaluate
            scores = [(self.calculate_cost(ind), ind) for ind in population]
            scores.sort(key=lambda x: x[0], reverse=True)
            
            best_score = scores[0][0]
            if gen % 10 == 0:
                print(f"Generation {gen}: Best Plan Profit = ${best_score:,.2f}")
                
            # Selection (Top 20%)
            top_performers = [x[1] for x in scores[:int(population_size*0.2)]]
            
            # Crossover & Mutation
            new_population = top_performers[:]
            while len(new_population) < population_size:
                parent1 = random.choice(top_performers)
                parent2 = random.choice(top_performers)
                
                # Crossover
                split = random.randint(0, self.fleet_size-1)
                child = parent1[:split] + parent2[split:]
                
                # Mutation (5% chance per gene)
                if random.random() < 0.3:
                    idx = random.randint(0, self.fleet_size-1)
                    new_s = random.randint(0, len(self.supply_nodes)-1)
                    new_d = random.randint(0, len(self.demand_nodes)-1)
                    child[idx] = (new_s, new_d)
                    
                new_population.append(child)
            
            population = new_population
            
        return scores[0][0], scores[0][1]

if __name__ == "__main__":
    system = AgriFlowV2()
    best_profit, best_plan = system.run_genetic_algorithm()
    print(f"\nFINAL RESULT:")
    print(f"Optimal Static Plan Profit: ${best_profit:,.2f}")
