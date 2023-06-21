from config import *
from models import ReplayBuffer, QNetwork
import torch.optim as optim
import numpy as np
import random
import torch
from torch.nn.functional import mse_loss


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON

        self.memory = ReplayBuffer(capacity=BUFFER_SIZE)

        self.model = QNetwork(self.state_size, self.action_size)
        self.target_model = QNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy())

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(BATCH_SIZE)

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.FloatTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.BoolTensor(done_batch)

        x = self.model(state_batch)

        current_q_values = self.model(state_batch).gather(0, action_batch.long().unsqueeze(1))
        next_q_values = self.target_model(next_state_batch).gather(0, action_batch.long().unsqueeze(1))
        target_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch.long())

        loss = mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
