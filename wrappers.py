from pettingzoo.butterfly import knights_archers_zombies_v10
from supersuit.utils.base_aec_wrapper import PettingzooWrap
import numpy as np
from config import *
from supersuit.lambda_wrappers import reward_lambda_v0


def reward_shaping(obs):
    second_lowest_dist = np.partition(obs[1:(KNIGHTS + ARCHERS + 1), 0],  1)[1]
    return min(second_lowest_dist, 0.3) * REWARD_SHAPING_FACTOR


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
        # self.__cumulative_rewards = {a: 0 for a in self.agents}
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