from collections import deque
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any
from random import sample, random
import wandb
import numpy as np
from tqdm import tqdm
import time
from kaggle_environments import make, Environment, environments
from copy import deepcopy


@dataclass
class SARSD:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    def __init__(self, buffer_size=1_000_000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def insert(self, sars):
        self.buffer.append(sars)

    def sample(self, num_samples):
        assert num_samples <= len(self.buffer)
        return sample(self.buffer, num_samples)


class Model(nn.Module):
    def __init__(self, obs_shape, num_actions, lr=0.00001):
        super(Model, self).__init__()
        self.obs_shape, self.num_actions = obs_shape, num_actions
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape[0], 128),
            torch.nn.Linear(128, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions), torch.nn.Tanh()
        )
        self.initialize_layers()

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def initialize_layers(self):
        # Initialize layers
        for layer in self.net[1:-2:2]:
            torch.nn.init.kaiming_normal_(layer.weight)
        torch.nn.init.normal_(self.net[-2].weight)

    def forward(self, x):
        return self.net(x)


def pre_process(observation):
    flat_board = np.zeros(77)
    geese = observation['geese']
    food = observation['food']
    index = observation['index']

    for i, goose in enumerate(geese):
        value = [1, 2][index == i]
        for i, el in enumerate(goose):
            flat_board[el] = value + [0., 0.5][i == 0]  # Different number if head

    for el in food:
        flat_board[el] = -1

    return flat_board.reshape(7, 11)


class Agent:
    def __init__(self, model):
        self.model = model
        self.target = deepcopy(model)

    def get_action(self, observation, epsilon):
        prediction = self.model(torch.Tensor([observation]))[0].detach().numpy()
        num_actions = self.model.num_actions

        if np.random.random() < epsilon:
            available_actions = [c for i, c in enumerate(range(num_actions)) if observation[i] == 0]
            return int(np.random.choice(available_actions)), np.max(prediction)
        else:
            for i in range(num_actions):
                if observation[i] != 0:
                    prediction[i] = -1
            return int(np.argmax(prediction)), np.max(prediction)

    def update_target_model(self):
        self.target.load_state_dict(self.model.state_dict())

    def save_model_to_disk(self, path):
        torch.save(self.model.state_dict(), path)

    def train_step(self, state_transitions, num_actions, device, discount_factor=0.99):
        cur_states = torch.stack([torch.Tensor(s.state) for s in state_transitions]).to(device)
        rewards = torch.stack([torch.Tensor([s.reward]) for s in state_transitions]).to(device)
        mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions]).to(device)
        next_states = torch.stack([torch.Tensor(s.next_state) for s in state_transitions]).to(device)
        actions = [s.action for s in state_transitions]

        move_validity = next_states[:, :num_actions] == 0
        with torch.no_grad():
            q_values_next = self.target(next_states)
        q_values_next = -np.where(move_validity, q_values_next, -1).max(-1)

        self.model.optimizer.zero_grad()
        qvals = self.model(cur_states)
        one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

        actual_values = rewards[:, 0] + mask[:, 0] * q_values_next * discount_factor

        expected_values = torch.sum(qvals * one_hot_actions, -1)

        loss = ((actual_values - expected_values) ** 2).mean()

        loss.backward()
        self.model.optimizer.step()

        return loss


class ConnectX(Environment):
    def __init__(self):
        super(ConnectX, self).__init__(**environments['connectx'])
        self.action_space = gym.spaces.Discrete(self.configuration.columns)
        self.observation_space = np.array([0] * self.configuration.columns * self.configuration.rows)


class Trainer:
    def __init__(self, test=False, checkpoint=None, device='cpu', min_rb_size=100_000, sample_size=4_096,
                 eps=1, eps_min=0.1, eps_decay=0.999999, env_steps_before_train=64, tgt_model_update=250):
        if not test:
            wandb.init(project="dqn-tutorial", name="dqn-minimax")

        self.tq = tqdm()

        self.min_rb_size = min_rb_size
        self.sample_size = sample_size

        self.test = test
        self.checkpoint = checkpoint
        self.device = device

        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        self.env_steps_before_train = env_steps_before_train
        self.tgt_model_update = tgt_model_update

        self.env = make("hungry_geese", configuration={"columns": 11, "rows": 7, "hunger_rate": 40, "min_food": 2},
                        debug=True)

        self.last_observation = self.env.reset()[0]['observation']

        print(self.last_observation)

        self.last_observation = pre_process(observation=self.last_observation)

        obs_space = (7, 11)

        model = Model(obs_space, 4).to(device)
        if checkpoint is not None:
            print('Models loaded:')
            model.load_state_dict(torch.load(checkpoint))

        self.agent = Agent(model=model)

        self.rb = ReplayBuffer()
        self.steps_since_train = 0
        self.epochs_since_tgt = 0

        self.step_num = -1 * self.min_rb_size

        self.episode_rewards = []

        self.rolling_reward = []
        self.active_player = 0

    def play(self):

        if self.test:
            self.env.render()
            time.sleep(0.01)

        self.tq.update(1)
        self.eps = max(self.eps_decay ** self.eps, self.eps_min)
        if self.test:
            self.eps = 0

        action, prediction = self.agent.get_action(observation=self.last_observation, epsilon=self.eps)

        observation, reward, done = self.process_action(action)

        self.rb.insert(SARSD(self.last_observation, action, reward, observation, done))

        self.finish_step_routine(observation=observation, prediction=prediction)

        if done:
            self.new_game()

        if not self.test and self.step_num > self.min_rb_size and self.steps_since_train > self.env_steps_before_train:
            self.train_model_routine()

            if self.epochs_since_tgt > self.tgt_model_update:
                self.update_target_model_routine()

    def switch_active_player(self):
        self.active_player = [1, 0][self.active_player]

    def process_action(self, action):
        p_dict = self.env.step([action if i == self.active_player else None for i in [0, 1]])

        reward = p_dict[self.active_player]['reward']
        done = self.env.done

        observation = p_dict[[1, 0][self.active_player]]['observation']
        observation = pre_process(observation=observation)

        reward = 1 if reward == 1 else 0
        return observation, reward, done

    def new_game(self):
        self.episode_rewards.append(np.mean(self.rolling_reward))
        if self.test:
            print(self.rolling_reward)
        self.rolling_reward = []
        self.last_observation = pre_process(observation=self.env.reset()[0]['observation'])
        self.active_player = 0

    def update_target_model_routine(self):
        self.agent.update_target_model()
        self.agent.save_model_to_disk(f'models/{self.step_num}.pth')
        self.epochs_since_tgt = 0

    def train_model_routine(self):
        self.episode_rewards = []
        self.epochs_since_tgt += 1
        self.steps_since_train = 0
        loss = self.agent.train_step(self.rb.sample(self.sample_size), self.env.action_space.n, self.device)
        self.update_wandb(loss=loss)

    def update_wandb(self, loss):
        wandb.log({'loss': loss.detach().cpu().item(),
                   'eps': self.eps,
                   'rewards': np.mean(self.episode_rewards)
                   },
                  step=self.step_num)

    def finish_step_routine(self, observation, prediction):
        self.rolling_reward.append(prediction)
        self.switch_active_player()
        self.last_observation = observation
        self.steps_since_train += 1
        self.step_num += 1


def main(test=False, checkpoint=None, device='cpu'):
    game = Trainer(test=test, checkpoint=checkpoint, device=device)
    try:
        while True:
            game.play()

    except KeyboardInterrupt:
        pass
