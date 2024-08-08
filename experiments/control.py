import math
import seaborn as sns

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# TODO: solve CartPole like for (actions)
# - Throttle: [0, 100] percent
# - Aileron
# - Elevator
# Or mujoco like for all three combined # (?)

env = gym.make("CartPole-v1")
# env = gym.make("mujoco") # (?)

# Q-learning with shallow neural network as function approximator.
# \delta = Q(s, a) - (r - \gamma max_{a}^{'} Q(s^{'}, a))
# where \delta is temporal difference error and 
# \gamma is the discount factor.
# [DQN] https://doi.org/10.48550/arXiv.1312.5602
# [Wikipedia] https://en.wikipedia.org/wiki/Q-learning
# [Found Survey Paper] https://doi.org/10.48550/arXiv.2304.00026

# State Space:
# Speed (Throttle): [0, 1000 or INF]
# Bank Angle (Aileron): [-180, +180] degrees
# Pitch (Elevator): [-90, 90] degrees
# State and Action space are continuous.

class QN(nn.Module):
    layer_sizes = [32, 64, 32]
    
    # Single variable => num_states = 1 and num_actions = 1
    # All variables => num_states = 3 (speed, bank angle, pitch), 
    # num_actions = 3 (throttle, aileron, elevator)
    def __init__(self, num_states: int, num_actions: int):
        super(QN, self).__init__()

        self.layers = list({
            nn.Linear(num_states, self.layer_sizes[0])
        })

        for i in range(1, len(self.layer_sizes)):
            self.layers.append(nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i]))

        self.layers.append(nn.Linear(self.layer_sizes[-1], num_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

# TODO: add trainer:
# - select or define loss function
# - select or define optimizer
# - create a simulation function or record data from actual flight
# - select action using epsilon greedy approach to feed to sim

# Constants
GAMMA = 1 # discount factor
COMMENCE_EPSILON = 1
DECAY_EPSILON = 1
CULMINATE_EPSILON = 1

LEARNING_RATE = 1

# TODO

if __name__ == "__main__":
    q = QN(3, 3)
    assert(len(q.layers) == len(q.layer_sizes) + 1)
