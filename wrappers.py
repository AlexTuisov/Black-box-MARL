from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper


class RewardShapedEnv(BaseWrapper):


    def step(self, action) -> None:
        self.env.step(action)

        self.agent_selection = self.env.agent_selection
        # here the shaping takes place
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos
        self.agents = self.env.agents
        self._cumulative_rewards = self.env._cumulative_rewards


