import torch
import torch.nn as nn
import torch.nn.functional as F

class AgriFlowBrain(nn.Module):
    def __init__(self, hidden_size):
        super(AgriFlowBrain, self).__init__()
        self.fc1 = nn.Linear(15, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.actor = nn.Linear(hidden_size // 2, 5) 
        self.critic = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = F.elu(self.fc1(x)) 
        x = F.elu(self.fc2(x))
        return F.softmax(self.actor(x), dim=-1), self.critic(x)