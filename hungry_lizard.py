import numpy as np

from gym import spaces
from gym.utils import seeding
from numpy import random
import pandas as pd

import gym


class HungryLizard(gym.Env):
    """
    A environment to model a hungry lizard
        
    """

    def __init__(self,
                 w=4,
                 h=4,
                 birds_loc=None,
                 small_reward_loc=None,
                 large_reward_loc=None):
        self.width = w
        self.height = h
        self.area = int(self.width * self.height)
        self.observation_space = spaces.Discrete(self.area)

        self.action_space = spaces.Discrete(4)

        self.counter = 0

        self.state = None
        self.done = False

        self.current_step = 0
        self.reset()
        self.array = np.array([[i // self.width,
                                i % self.width] for i in range(self.area)])
        self.map = pd.DataFrame({'ind': np.arange(len(self.array)),
                                 'x': self.array[:, 1],
                                 'y': self.array[:, 0]})
        if birds_loc is None:
            self.birds_loc = [9, 6, 3]
        else:
            self.birds_loc = birds_loc
        if small_reward_loc is None:
            self.small_reward_loc = [5, 2, 13, 10, 7]
        else:
            self.small_reward_loc = np.array(small_reward_loc)
        if large_reward_loc is None:
            self.large_reward_loc = [15]
        else:
            self.large_reward_loc = np.array(large_reward_loc)

    def reset(self):
        self.state = 0
        self.done = False
        return None

    def step(self, action):
        x = self.map[self.map['ind'] == self.state]['x'].values[0]
        y = self.map[self.map['ind'] == self.state]['y'].values[0]
        if action == 0:
            new_x = x
            new_y = np.clip(y + 1, 0, self.height - 1)
        elif action == 1:
            new_x = np.clip(x + 1, 0, self.width - 1)
            new_y = y
        elif action == 2:
            new_x = x
            new_y = np.clip(y - 1, 0, self.height - 1)
        elif action == 3:
            new_x = np.clip(x - 1, 0, self.width - 1)
            new_y = y
        else:
            raise (Exception('Action must be integer between 0 and 3'))
        new_state = self.map[(self.map['x'] == new_x)
                             & (self.map['y'] == new_y)]['ind'].values[0]
        if not self.done:
            self.state = new_state
            reward, done = self.compute_reward()
        else:
            self.state = self.state
            reward = 0
            done = True
        self.done = done

        info = None

        return self.state, reward, done, info

    def compute_reward(self):
        if self.state in self.birds_loc:
            reward = -100
            done = True
        elif self.state in self.small_reward_loc:
            self.small_reward_loc.remove(self.state)
            reward = 3
            done = False
        elif self.state in self.large_reward_loc:
            reward = 10
            done = True
        else:
            reward = -1
            done = False
        return reward, done

    def render(self, mode='human'):
        print("Not implemented")
        return

    def close(self):
        print("Not implemented")
        return
