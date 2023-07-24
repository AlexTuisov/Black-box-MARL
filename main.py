from pettingzoo.butterfly import knights_archers_zombies_v10
# from gym.spaces.discrete import Discrete
from config import *
from stable_baselines3.ppo import CnnPolicy,MlpPolicy
from stable_baselines3 import PPO
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss
from pettingzoo.test import parallel_api_test
if WANDB:
    import wandb

    wandb.init(
        # set the wandb project where this run will be logged
        project="MAExplanations",

        # track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "EPSILON_MIN": EPSILON_MIN,
            "HIDDEN_DQN_SIZE": HIDDEN_DQN_SIZE
        }
    )


def define_environment():
    env = knights_archers_zombies_v10.env(
        spawn_rate=SPAWN_RATE,
        num_archers=0,
        num_knights=4,
        max_zombies=10,
        max_arrows=10,
        pad_observation=True,
        render_mode="human",
        vector_state=True
        )

    env = ss.frame_stack_v1(env, 3)
    env = ss.black_death_v3(env)
    env = aec_to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')
    print(env.action_space)
    return env

def main():
    env = define_environment()
    model = PPO(MlpPolicy, env, verbose=1, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211,
                vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)

    model.learn(total_timesteps=200000)
    model.save('policy')


if __name__ == '__main__':
    main()
