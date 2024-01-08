from dataclasses import dataclass, astuple
from scipy.optimize import basinhopping, dual_annealing, brute
import numpy as np
from config import *
from utils import *
from icecream import ic

@dataclass
class Model:
    param1: float
    param2: float

    def to_array(self):
        return astuple(self)


def model_eval(model: Model):
    # Replace this with your actual model evaluation logic
    env = define_base_environment(visual=False)
    param1 = model.param1
    param2 = model.param2
    knight_model = PPO.load(SINGLE_KNIGHT_FILE_PATH)
    rewards = {agent: 0 for agent in env.possible_agents}
    bounds = [[0, 0] for agent in env.possible_agents]
    for agent_index, bound in enumerate(bounds):
        if agent_index == 0:
            bounds[agent_index] = 0, param1 + MARGIN
        elif agent_index == 1:
            bounds[agent_index] = param1 - MARGIN, param2 + MARGIN
        else:
            bounds[agent_index] = param2 - MARGIN, 1
    for _ in range(NUMBER_OF_GAMES_TO_CHECK_MODEL):
        env.reset()
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            for an_agent in env.agents:
                rewards[an_agent] += env.rewards[an_agent]
            agent_index = env.agents.index(agent)
            if not np.any(obs) and truncation:
                act = None
            elif "knight" in agent:
                act = knight_model.predict(transform_array_to_single_knight(obs, bounds[agent_index][0], bounds[agent_index][1]),
                                           deterministic=False)[0]
            env.step(act)
    value = sum(rewards.values()) / NUMBER_OF_GAMES_TO_CHECK_MODEL
    ic(rewards)
    ic(value)
    ic(model.param1)
    ic(model.param2)
    ic(bounds)
    print("-------------")
    return -value


def basinhopping_wrapper(x):
    model = Model(*x)
    return model_eval(model)


def main():
    initial_model = Model(0.05, 0.15)  # Update with your actual initial guess

    # Convert the initial model to an array for basinhopping
    initial_guess = initial_model.to_array()

    # initial_guess = np.ndarray([0.2, 0.85], dtype=float)

    # Minimization using annealing
    result = brute(basinhopping_wrapper, ranges=((0, 0.5), (0, 0.7)), Ns=11, finish=None)
    # result = basinhopping(basinhopping_wrapper, initial_guess, niter=3)
    # result = dual_annealing(basinhopping_wrapper, [[0, 1], [0, 1]], maxiter=10)
    ic(result)

    # Extract the best fit model

    print("Function value:", result.fun)


if __name__ == '__main__':
    main()
