import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import osmnx as ox
from tqdm import tqdm

from config import *
from environment import AgriFlowEnv
from cortex import AgriFlowBrain

def train():
    env = AgriFlowEnv()
    agent = AgriFlowBrain(hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    
    print("Engaging Receding Horizon Training...")
    
    for episode in tqdm(range(EPISODES)): 
        state = env.reset()
        state = torch.FloatTensor(state).to(DEVICE)
        
        log_probs, values, rewards = [], [], []
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            action_probs, value = agent(state)
            
            # Sample Action
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            # Step Environment
            next_state, reward, done, _, _ = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            
            state = torch.FloatTensor(next_state).to(DEVICE)
            steps += 1
        
        # Update (PPO-style Advantage)
        if len(rewards) > 0: # Check to ensure we actually took steps
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + GAMMA * R
                returns.insert(0, R)
            returns = torch.tensor(returns).to(DEVICE)
            
            # Normalize returns only if there is variance
            if returns.std() > 1e-9:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            loss = 0
            for log_prob, value, R in zip(log_probs, values, returns):
                advantage = R - value.item()
                loss += -log_prob * advantage + F.smooth_l1_loss(value, torch.tensor([R]).to(DEVICE).unsqueeze(0))
                
            optimizer.zero_grad()
            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()
            
        if (episode + 1) % 10 == 0:
            avg_reward = sum(rewards) if rewards else 0
            print(f"Episode {episode+1}/{EPISODES} | Reward: {avg_reward:.2f} | Steps: {steps}")

    torch.save(agent.state_dict(), MODEL_FILENAME)
    return agent, env

def visualize(agent, env):
    state = env.reset()
    state = torch.FloatTensor(state).to(DEVICE)
    path = [env.current_node]
    done = False
    
    print("Computing Optimal Spoilage-Aware Path...")
    steps = 0
    
    # CRITICAL FIX: Use the environment's step function instead of manual logic
    # This ensures the state tensor is updated correctly for the Neural Net
    while not done and steps < MAX_STEPS * 2: # Allow double steps for vis
        with torch.no_grad():
            action_probs, _ = agent(state)
            # Deterministic choice for visualization (Best action)
            action = torch.argmax(action_probs).item()
        
        next_state, _, done, _, _ = env.step(action)
        
        # Track path
        if env.current_node != path[-1]:
            path.append(env.current_node)
        
        state = torch.FloatTensor(next_state).to(DEVICE)
        steps += 1

    print(f"Path Found! Length: {len(path)} nodes.")
    
    # Dark Mode Visualization
    fig, ax = ox.plot_graph_route(env.G, path, route_linewidth=6, node_size=0, 
                                bgcolor='#111111', edge_color='#333333', route_color='#00ffcc', 
                                show=False, close=False)
    
    # Add Start/End Markers
    start_node = path[0]
    end_node = path[-1]
    start_xy = (env.G.nodes[start_node]['x'], env.G.nodes[start_node]['y'])
    end_xy = (env.G.nodes[end_node]['x'], env.G.nodes[end_node]['y'])
    
    ax.scatter(start_xy[0], start_xy[1], c='#00FF00', s=200, marker='o', label='Start', zorder=10)
    ax.scatter(end_xy[0], end_xy[1], c='#FF0000', s=300, marker='*', label='End', zorder=10)
    
    plt.savefig(IMAGE_FILENAME)
    print(f"Visualization saved to {IMAGE_FILENAME}")

if __name__ == "__main__":
    agent, env = train()
    visualize(agent, env)