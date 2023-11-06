from dataclasses import dataclass, astuple
from scipy.optimize import basinhopping
import numpy as np
from config import *
from utils import *


@dataclass
class Model:
    param3: list

    # Add more parameters if needed

    def to_array(self):
        return self.param3


def model_eval(model: Model):
    # Replace this with your actual model evaluation logic
    env = define_base_environment(visual=False)
    knight_model = PPO.load(SINGLE_KNIGHT_FILE_PATH)
    rewards = {agent: 0 for agent in env.possible_agents}
    for _ in range(NUMBER_OF_GAMES_TO_CHECK_MODEL):
        env.reset()
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            for agent in env.agents:
                rewards[agent] += env.rewards[agent]
            agent_index = env.agents.index(agent)
            lower_bound, higher_bound = model.param3[agent_index] - MARGIN, model.param3[agent_index + 1] + MARGIN
            if not np.any(obs) and truncation:
                act = None
            elif "knight" in agent:
                act = knight_model.predict(transform_array_to_single_knight(obs, lower_bound, higher_bound),
                                           deterministic=False)[0]

            env.step(act)
    return sum(rewards.values()) / len(rewards.values())


def basinhopping_wrapper(x):
    model = Model(x)
    return model_eval(model)


def main():
    # Define the initial model
    initial_model = Model([0, 0.1, 0.2, 0.8, 1])  # Update with your actual initial guess

    # Convert the initial model to an array for basinhopping
    initial_guess = initial_model.to_array()

    # Minimization using basinhopping
    result = basinhopping(basinhopping_wrapper, initial_guess, niter=3)

    # Extract the best fit model
    best_model = Model(result.x)

    print("Best model found:", best_model)
    print("Function value:", result.fun)


if __name__ == '__main__':
    main()
