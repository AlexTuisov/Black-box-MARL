from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper


class RewardShapedEnv(BaseWrapper):

    def step(self, action):
        super().step(action)
        # obs, reward, termination, truncation, info = self.env.last()
        # modified_reward = self.reward_shaping(obs, reward)



