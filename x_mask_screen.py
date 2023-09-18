from config import *
import numpy as np


def check_param(space, param):
    pass


def change_obs_space(obs_space, param):
    return obs_space


def change_observation(obs, obs_space, param):
    new_obs = np.zeros_like(obs)
    new_obs[:-ZOMBIES, :] = obs[:-ZOMBIES, :]
    mask = ((obs[-ZOMBIES:, 1] + obs[0, 1]) > param["lower_bound"]) & (
                (obs[-ZOMBIES:, 1] + obs[0, 1]) <= param["higher_bound"])
    new_obs[-ZOMBIES:, :] = obs[-ZOMBIES:, :] * mask[:, np.newaxis]
    return new_obs
