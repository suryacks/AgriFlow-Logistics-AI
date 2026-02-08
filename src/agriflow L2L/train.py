import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os

from config import *
from environment import AgriFlowEnv
from cortex import AgriFlowBrain

def train_system():
    env = AgriFlowEnv()
    agent = AgriFlowBrain(hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
    
    print(f"Training AgriFlow Cortex on {DEVICE}...")
    
    # Check if model exists to resume (Optional)
    # if os.path.exists(MODEL_FILENAME):
    #     agent.load_state_dict(torch.load(MODEL_FILENAME))
    
    for episode in tqdm(range(EPISODES)): 
        # Randomize mission every 50 episodes to prevent overfitting
        if episode % 50 == 0:
            env.set_random_mission()
            
        state = env.reset()
        state = torch.FloatTensor(state).to(DEVICE)
        
        log_probs, values, rewards = [], [], []
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            action_probs, value = agent(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            next_state, reward, done, _, _ = env.step(action.item())
            
            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)
            
            state = torch.FloatTensor(next_state).to(DEVICE)
            steps += 1
        
        # PPO Update Logic
        if len(rewards) > 0:
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + GAMMA * R
                returns.insert(0, R)
            returns = torch.tensor(returns).to(DEVICE)
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

    print("Training Complete. Saving Cortex...")
    torch.save(agent.state_dict(), MODEL_FILENAME)

if __name__ == "__main__":
    train_system()