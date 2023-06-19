from pettingzoo.butterfly import knights_archers_zombies_v10
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def main():
    # Create an instance of the environment to get the list of agents
    env_for_agents = knights_archers_zombies_v10.env(render_mode="human")
    env_for_agents.reset()
    agents = env_for_agents.agents

    envs_and_models = []
    for agent in agents:
        envs_and_models.append

    # # Create a separate model and environment instance for each agent
    # envs_and_models = {
    #     agent: (
    #         knights_archers_zombies_v10.env(render_mode="human"),
    #         PPO("MlpPolicy", knights_archers_zombies_v10.env(), verbose=1)
    #     )
    #     for agent in agents
    # }
    #
    # for i in range(10000):  # adjust as needed
    #     for agent, (env, model) in envs_and_models.items():
    #         env.reset()
    #         for _ in range(1000):  # adjust as needed
    #             obs, reward, done, info = env.last()
    #             action, _states = model.predict(obs)
    #             env.step(action)
    #             if not done:
    #                 model.learn(total_timesteps=1)
    #             else:
    #                 break

    for agent, (env, model) in envs_and_models.items():
        model.save(f"ppo_kaz_{agent}")



if __name__ == '__main__':
    main()
