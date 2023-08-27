from pettingzoo.butterfly import knights_archers_zombies_v10
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from config import *
import numpy as np
from stable_baselines3.ppo import CnnPolicy,MlpPolicy
from stable_baselines3 import PPO, DQN
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss
from stable_baselines3.common.vec_env import VecTransposeImage
from wrappers import *


def define_base_environment(visual):
    render_mode = "human" if visual else None
    env = knights_archers_zombies_v10.env(
        spawn_rate=SPAWN_RATE,
        num_archers=ARCHERS,
        num_knights=KNIGHTS,
        max_zombies=ZOMBIES,
        max_arrows=2*ARCHERS,
        pad_observation=True,
        render_mode=render_mode,
        vector_state=VECTOR_INPUT,
        max_cycles=2400
    )
    if not VECTOR_INPUT:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
    env = RewardShapedEnv(env)
    env = ss.frame_stack_v1(env, 3)
    env = ss.black_death_v3(env)

    return env

def define_environment_for_training():
    env = define_base_environment(visual=False)
    env = aec_to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')
    env = ss.concat_vec_envs_v1(env, 4, num_cpus=4, base_class='stable_baselines3')
    return env

def define_environment_for_playing():
    env = define_base_environment(visual=True)
    return env

def train():
    env = define_environment_for_training()
    eval_env = define_environment_for_training()
    if not VECTOR_INPUT:
        eval_env = VecTransposeImage(eval_env)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=10, verbose=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/", verbose=1,
                                 log_path="./logs/", eval_freq=10000, callback_after_eval=stop_train_callback,
                                 n_eval_episodes=10, deterministic=True, render=False)
    if ppo:
        model = PPO(MlpPolicy if VECTOR_INPUT else CnnPolicy,
                    env,
                    verbose=0,
                    gamma=GAMMA,
                    learning_rate=LEARNING_RATE,
                    ent_coef=ENTROPY_COEFFICIENT,
                    n_epochs=NUM_EPOCHS,
                    n_steps=NUM_STEPS
                )
    else:
        model = DQN("MlpPolicy" if VECTOR_INPUT else "CnnPolicy", env, verbose=3)
    model.learn(total_timesteps=FRAMES_TO_LEARN, callback=eval_callback)
    model = PPO.load("./logs/best_model.zip")
    model.save(FILE_PATH)

def load_policy():
    env = define_environment_for_playing()
    model = PPO.load(FILE_PATH)
    for _ in range(5):
        env.reset()
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if not np.any(obs) and truncation:
                act = None
            else:
                act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    return model


def main():
    if TRAIN:
        train()
        training_result = np.load(FILE_PATH)
        for result in training_result:
            print(result)
    load_policy()



if __name__ == '__main__':
    main()
