import json
import math
import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Optimizer
from tqdm import tqdm


class HDQNEnvironment(ABC):
    def __init__(self, num_actions: int, state_size: int, num_goals: int):
        self._num_actions = num_actions
        self._state_size = state_size
        self._num_goals = num_goals

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def state_size(self) -> int:
        return self._state_size

    @property
    def num_goals(self) -> int:
        return self._num_goals

    @abstractmethod
    def reset(self) -> np.ndarray:
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        pass

    @abstractmethod
    def achieved_goal(self, goal: int, state: np.ndarray) -> bool:
        pass


class LunarLander(HDQNEnvironment):
    def __init__(self,
                 num_goals: int = 6,
                 reward_scaling: float = 200.0,
                 clip_reward: bool = True,
                 render: bool = True):
        self._reward_scaling = reward_scaling
        self._clip_reward = clip_reward
        self._render = render
        self._env = gym.make('LunarLander-v2')
        self._env.reset()
        super().__init__(self._env.action_space.n, self._env.observation_space.shape[0], num_goals)

    def reset(self) -> np.ndarray:
        return self._env.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self._render:
            self._env.render()

        next_state, reward, done, _ = self._env.step(action)

        norm_reward = reward / self._reward_scaling
        if self._clip_reward:
            norm_reward = max(norm_reward, -1.0)
            norm_reward = min(norm_reward, 1.0)

        return next_state, norm_reward, done

    def achieved_goal(self, goal: int, state: np.ndarray) -> bool:
        if goal == 0:  # Left leg touches
            return state[6] == 1.0
        elif goal == 1:  # Right leg touches
            return state[7] == 1.0
        elif goal == 2:  # Within ten percent of X coordinate
            return abs(state[0]) < 0.10
        elif goal == 3:  # Within ten percent of Y coordinate
            return abs(state[1]) < 0.10
        elif goal == 4:  # Angular rotation is less than 0.10
            return abs(state[5]) < 0.10
        elif goal == 5:  # Y velocity is less than 0.10
            return abs(state[3]) < 0.10
        else:
            return False


class StochasticMDP(HDQNEnvironment):
    def __init__(self, state_size: int = 6, prob_right: float = 0.5):
        num_goals = state_size
        super().__init__(2, state_size, num_goals)
        self._prob_right = prob_right
        self._current_state = 2
        self._visited_last_state = False

    def reset(self) -> np.ndarray:
        self._current_state = 2
        self._visited_last_state = False
        state = create_one_hot(self._current_state - 1, self.state_size)
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self._current_state != 1:
            if action == 1:
                if self._current_state < self.state_size and random.random() < self._prob_right:
                    self._current_state += 1
                else:
                    self._current_state -= 1
            elif action == 0:
                self._current_state -= 1

            if self._current_state == self.state_size:
                self._visited_last_state = True

        state = create_one_hot(self._current_state - 1, self.state_size)

        if self._current_state == 1:
            if self._visited_last_state:
                return state, 1.00, True
            else:
                return state, 0.01, True
        else:
            return state, 0.0, False

    def achieved_goal(self, goal: int, state: np.ndarray) -> bool:
        return goal == np.argmax(state)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self._capacity = capacity
        self._buffer = deque(maxlen=self._capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        state, action, reward, next_state, done = zip(*random.sample(self._buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self) -> int:
        return len(self._buffer)


class Net(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int, hidden_units: int = 256):
        super(Net, self).__init__()

        self._layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, num_outputs))

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self._layers(x)

    def act(self, state: np.ndarray, epsilon: float, num_actions: int) -> int:
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            action = self.forward(Variable(state, volatile=True)).max(1)[1]  # TODO(christhetree): fix this
            return int(action.data[0])  # TODO(christhetree): fix this
        else:
            return random.randrange(num_actions)


def create_one_hot(idx: int, length: int) -> np.ndarray:
    one_hot = np.zeros(length)
    one_hot[idx] = 1.0
    return one_hot


def update(model: Net,
           optimizer: Optimizer,
           replay_buffer: ReplayBuffer,
           batch_size: int,
           gamma: float = 0.99) -> None:
    if batch_size > len(replay_buffer):
        return

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(state))
    next_state = Variable(
        torch.FloatTensor(next_state),
        volatile=True)  # TODO(christhetree): fix this
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_value = model(state)
    q_value = q_value.gather(1, action.unsqueeze(1)).squeeze(1)

    next_q_value = model(next_state).max(1)[0]
    expected_q_value = reward + (gamma * next_q_value * (1 - done))

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # TODO(christhetree): fix this


def anneal_epsilon_exp(frame_idx: int,
                       total_num_frames: int,
                       epsilon_start: float,
                       epsilon_final: float,
                       decay_factor: float = 5.0) -> float:
    if frame_idx > total_num_frames:
        return epsilon_final

    return epsilon_final + ((epsilon_start - epsilon_final) * math.exp(
        -float(frame_idx) / (total_num_frames / decay_factor)))


def anneal_epsilon_linear(frame_idx: int, total_num_frames: int,
                          epsilon_start: float, epsilon_final: float) -> float:
    if frame_idx > total_num_frames:
        return epsilon_final

    return epsilon_start - ((frame_idx / float(total_num_frames)) * (epsilon_start - epsilon_final))


def run():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    total_num_frames = 50000
    save_interval_frames = None
    plot_interval_frames = total_num_frames
    num_eps_per_point = 100
    epsilon_start = 1.0
    epsilon_final = 0.01
    anneal_epsilon_fn = anneal_epsilon_exp
    batch_size = 32
    buffer_capacity = 100000

    unique_identifier = 'testing'
    load_model_path = ''
    load_meta_model_path = ''
    load_episode_rewards_path = ''

    env = StochasticMDP()
    # env = LunarLander(render=True)

    # TODO(christhetree): implement target networks
    if load_model_path:
        model = torch.load(load_model_path)
    else:
        model = Net(env.state_size + env.num_goals, env.num_actions)

    # TODO(christhetree): implement target networks
    if load_meta_model_path:
        meta_model = torch.load(load_meta_model_path)
    else:
        meta_model = Net(env.state_size, env.num_goals)

    optimizer = optim.Adam(model.parameters())
    meta_optimizer = optim.Adam(meta_model.parameters())

    replay_buffer = ReplayBuffer(buffer_capacity)
    meta_replay_buffer = ReplayBuffer(buffer_capacity)

    episode_rewards = []
    if load_episode_rewards_path:
        with open(load_episode_rewards_path, 'r') as episode_rewards_file:
            episode_rewards = json.load(episode_rewards_file)

    current_state = env.reset()
    done = False
    episode_reward = 0.0

    frame_idx = 1
    progress_bar = tqdm(total=total_num_frames)

    while frame_idx < total_num_frames:
        goal = meta_model.act(
            current_state,
            anneal_epsilon_fn(frame_idx, total_num_frames, epsilon_start, epsilon_final), env.num_goals)
        one_hot_goal = create_one_hot(goal, env.num_goals)

        meta_state = current_state
        extrinsic_reward = 0.0

        while not done and not env.achieved_goal(goal, current_state):
            goal_state = np.concatenate([current_state, one_hot_goal])
            action = model.act(
                goal_state,
                anneal_epsilon_fn(frame_idx, total_num_frames, epsilon_start, epsilon_final), env.num_actions)
            next_state, reward, done = env.step(action)

            episode_reward += reward
            extrinsic_reward += reward

            if env.achieved_goal(goal, next_state):
                intrinsic_reward = 1.0
            else:
                intrinsic_reward = 0.0

            replay_buffer.push(goal_state, action, intrinsic_reward,
                               np.concatenate([next_state, one_hot_goal]),
                               done)
            current_state = next_state

            update(model, optimizer, replay_buffer, batch_size)
            update(meta_model, meta_optimizer, meta_replay_buffer, batch_size)

            progress_bar.update()
            frame_idx += 1

            if save_interval_frames and frame_idx % save_interval_frames == 0:
                torch.save(model, f'model_{unique_identifier}_{frame_idx}')
                torch.save(meta_model, f'meta_model_{unique_identifier}_{frame_idx}')
                with open(f'episode_rewards_{unique_identifier}_{frame_idx}.json', 'w') as episode_rewards_file:
                    json.dump(episode_rewards, episode_rewards_file)

            if plot_interval_frames and frame_idx % plot_interval_frames == 0:
                plt.figure(figsize=(20, 5))
                plt.title(f'{env.__class__.__name__} {frame_idx} Frames')
                plt.plot([
                    np.mean(episode_rewards[idx:idx + num_eps_per_point])
                    for idx in range(0, len(episode_rewards), num_eps_per_point)
                ])
                plt.show()

        meta_replay_buffer.push(meta_state, goal, extrinsic_reward, current_state, done)

        if done:
            current_state = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            done = False

    progress_bar.close()


if __name__ == '__main__':
    run()
