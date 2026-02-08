
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loader import ScenarioLoader
from environment import LogisticsEnv
import numpy as np
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'resources', 'logistics_agent.pth')

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x, mask=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        
        if mask is not None:
            # Set invalid actions to -infinity so Softmax makes them 0 probability
            logits = logits.masked_fill(~mask, float('-inf'))
            
        return F.softmax(logits, dim=-1)

def train_logistics_agent():
    # 1. Load Data
    loader = ScenarioLoader()
    graph = loader.load_scenario()
    
    if len(graph.nodes) == 0:
        print("CRITICAL ERROR: No data loaded. Cannot train.")
        return

    # 2. Init Env
    env = LogisticsEnv(graph)
    
    # 3. Init Agent
    input_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n # This is huge (548)
    policy = PolicyNet(input_dim, n_actions)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    
    print(f"--- TRAINING LOGISTICS AGENT (MASKED) ---")
    
    # 4. Training Loop
    episodes = 500
    
    for episode in range(episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False
        steps = 0
        
        while not done and steps < 200: # Prevent infinite loops
            state_t = torch.FloatTensor(state)
            
            # Get valid moves
            mask = env.get_valid_actions_mask()
            
            # Forward Pass with Mask
            try:
                action_probs = policy(state_t, mask)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
            except ValueError:
                # Fallback if mask blocks everything (shouldn't happen with fallback logic)
                action = torch.tensor(0) 
            
            next_state, reward, done, _, _ = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            rewards.append(reward)
            state = next_state
            steps += 1
            
        # Update Strategy (Policy Gradient)
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        if returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        if len(policy_loss) > 0:
            optimizer.zero_grad()
            policy_loss = torch.stack(policy_loss).sum()
            policy_loss.backward()
            optimizer.step()
        
        if episode % 50 == 0:
            total_reward = sum(rewards)
            print(f"Episode {episode} | Steps: {steps} | Profit: ${total_reward:.2f}")

    print("Training Complete. Saving Logistics Brain...")
    torch.save(policy.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train_logistics_agent()
