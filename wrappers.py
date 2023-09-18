from pettingzoo.butterfly import knights_archers_zombies_v10
from supersuit.utils.base_aec_wrapper import PettingzooWrap
import numpy as np
from config import *
import random
from supersuit.lambda_wrappers import reward_lambda_v0
from supersuit.generic_wrappers.basic_wrappers import basic_obs_wrapper
import y_mask_screen


def reward_shaping(obs):
    second_lowest_dist = np.partition(np.abs(obs[1:(KNIGHTS + ARCHERS + 1), 1]),  1)[1]
    absolute_distances_to_zombies = np.abs(obs[9 + 1 - ZOMBIES:, 0])
    no_zombies = False if np.max(absolute_distances_to_zombies) > 0 else True
    if no_zombies:
        dist_to_closest_zombie = 0
        y_distance = 0.5 - abs(obs[0, 2] - 0.5)
    else:
        y_distance = 0
        number_of_agent = np.argmin(np.abs(obs[1:(KNIGHTS + ARCHERS + 1), 0]))
        area_of_responsibility = [number_of_agent / 3, (number_of_agent + 1) / 3]
        if area_of_responsibility[0] < obs[0, 1] < area_of_responsibility[1]:
            second_lowest_dist = 0
        dist_to_closest_zombie = np.min(absolute_distances_to_zombies[absolute_distances_to_zombies > 0])
        zombies_x_coordinates = obs[9 + 1 - ZOMBIES:, 1]
        absolute_zombie_coordinates = zombies_x_coordinates[absolute_distances_to_zombies != 0] + obs[0, 1]
        if not np.any((absolute_zombie_coordinates >= area_of_responsibility[0]) &
                      (absolute_zombie_coordinates <= area_of_responsibility[1])):
            dist_to_closest_zombie = 0
    return (min(second_lowest_dist, 0.3) - min(dist_to_closest_zombie, 0.3) + y_distance * 0.2) * REWARD_SHAPING_FACTOR


class RewardShapingEnv(PettingzooWrap):
    def __init__(self, env):
        super().__init__(env)

    def _check_wrapper_params(self):
        pass

    def _modify_spaces(self):
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.rewards = {
            agent: 0
            for agent, reward in self.rewards.items()
        }
        self.__cumulative_rewards = {a: 0 for a in self.agents}
        self._accumulate_rewards()

    def step(self, action):
        agent = self.env.agent_selection
        super().step(action)
        self.rewards = {
            agent: reward + reward_shaping(self.env.observe(agent))
            for agent, reward in self.rewards.items()
        }
        # self.__cumulative_rewards[agent] = 0
        # self._cumulative_rewards = self.__cumulative_rewards
        self._accumulate_rewards()


def mask_screen(env):
    return basic_obs_wrapper(env, y_mask_screen, None)



