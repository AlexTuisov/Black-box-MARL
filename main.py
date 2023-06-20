from pettingzoo.butterfly import knights_archers_zombies_v10
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def main():
    # Create an instance of the environment to get the list of agents
    env_for_agents = knights_archers_zombies_v10.env(render_mode="human")
    env_for_agents.reset()
    agents = env_for_agents.agents
    models = []

    for agent in agents:
        pass
        # models.append(create_model)


if __name__ == '__main__':
    main()
