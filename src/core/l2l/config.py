import torch

# --- CONFIGURATION ---
# Default Test Route (Vernon Hills Core)
DEFAULT_START = "1001 Centurion Ln, Vernon Hills, IL 60061"
DEFAULT_END = "300 Marriott Dr, Lincolnshire, IL 60069"

# Training Settings
HIDDEN_SIZE = 128
LEARNING_RATE = 0.0005
GAMMA = 0.98
EPISODES = 500  # Train enough to learn the heuristic
MAX_STEPS = 150

# Evaluation Settings
TEST_ROUNDS = 20 # Number of random routes to compare against Google Maps

# System
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOURCES_DIR = os.path.join(BASE_DIR, 'resources')
VISUALIZATIONS_DIR = os.path.join(BASE_DIR, 'visualizations')

GRAPH_FILENAME = os.path.join(RESOURCES_DIR, "agriflow_metro.graphml")
MODEL_FILENAME = os.path.join(RESOURCES_DIR, "agriflow_cortex.pth")