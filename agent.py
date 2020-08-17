import numpy as np
import random
from collections import namedtuple, deque

from nn import QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.t_step = 0
        random.seed(seed)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states ,dones = experiences

        q_targets_next = self.target_q_network(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        q_expected = self.local_q_network(states).gather(1, actions)
        # TODO: look into the paper to find the best loss function with discrete state actions

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        return state



    @property
    def memory(self):
        return ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)

    @property
    def optimizer(self):
        return optim.Adam(self.local_q_network.parameters(), lr=LR)

    @property
    def target_q_network(self):
        return QNetwork(self.state_size, self.action_size, self.seed).to(device)

    @property
    def local_q_network(self):
        return QNetwork(self.state_size, self.action_size, self.seed).to(device)

    @staticmethod
    def hello():
        return "hello"


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):

        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.memory = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).astype(np.uint8).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
