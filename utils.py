import time
import numpy as np
from icecream import ic

from config import (
    COORDINATION, ZOMBIES, VECTOR_INPUT, SPAWN_RATE, ARCHERS, KNIGHTS, MAX_NO_IMPROVEMENT_EVALS,
    LEARNING_RATE, GAMMA, ENTROPY_COEFFICIENT, NUM_EPOCHS, NUM_STEPS, FRAMES_TO_LEARN, FILE_PATH
)

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.ppo import MlpPolicy, CnnPolicy

from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss
from wrappers import *


def transform_array_to_single_knight(input_array, lower_bound, higher_bound):
    output_array = np.zeros((6, 5))
    output_array[0, :] = input_array[0, :]
    output_array[1, 4] = input_array[0, 4]

    if COORDINATION:
        mask = ((input_array[-ZOMBIES:, 1] + input_array[0, 1]) > lower_bound) & ((input_array[-ZOMBIES:, 1] + input_array[0, 1]) <= higher_bound)
        output_array[-ZOMBIES:, :] = input_array[-ZOMBIES:, :] * mask[:, np.newaxis]
    else:
        output_array[-ZOMBIES:, :] = input_array[-ZOMBIES:, :]
    return output_array


def transform_array_to_single_archer(input_array, lower_bound):
    output_array = np.zeros((7, 5))
    output_array[0, :] = input_array[0, :]
    output_array[1, 4] = input_array[0, 4]

    if COORDINATION:
        mask = input_array[-ZOMBIES:, 2] + input_array[0, 2] > lower_bound
        output_array[-ZOMBIES:, :] = input_array[-ZOMBIES:, :] * mask[:, np.newaxis]
    else:
        output_array[-ZOMBIES:, :] = input_array[-ZOMBIES:, :]
    return output_array


def define_base_environment(visual):
    render_mode = "human" if visual else None
    if CREATE_DATASET:
        render_mode = "rgb_array"
    env = knights_archers_zombies_v10.env(
        spawn_rate=SPAWN_RATE,
        num_archers=ARCHERS,
        num_knights=KNIGHTS,
        max_zombies=ZOMBIES,
        max_arrows=2*ARCHERS,
        pad_observation=True,
        render_mode=render_mode,
        vector_state=VECTOR_INPUT,
        max_cycles=2400,
        killable_knights=False
    )
    if not VECTOR_INPUT:
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
    if REWARD_SHAPING:
        env = RewardShapingEnv(env)
    env = ss.black_death_v3(env)

    return env

def define_environment_for_training():
    env = define_base_environment(visual=False)
    if COORDINATION:
        env = mask_screen(env)
    env = aec_to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class='stable_baselines3')
    return env

def define_environment_for_playing():
    env = define_base_environment(visual=True)
    return env

def train():
    start = time.time()
    env = define_environment_for_training()
    eval_env = define_environment_for_training()
    if not VECTOR_INPUT:
        eval_env = VecTransposeImage(eval_env)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=MAX_NO_IMPROVEMENT_EVALS, min_evals=MAX_NO_IMPROVEMENT_EVALS, verbose=1)
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/", verbose=1,
                                 log_path="./logs/", eval_freq=10000, callback_after_eval=stop_train_callback,
                                 n_eval_episodes=12, deterministic=True, render=False)
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
        model = DQN("MlpPolicy" if VECTOR_INPUT else "CnnPolicy",
                    env,
                    verbose=0,
                    gamma=GAMMA,
                    learning_rate=LEARNING_RATE
                    )
    model.learn(total_timesteps=FRAMES_TO_LEARN, callback=eval_callback)
    model = PPO.load("./logs/best_model.zip")
    model.save(FILE_PATH)
    finish = time.time()
    print(f"Total training time: {finish - start}")

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
                act = model.predict(obs, deterministic=False)[0]
            env.step(act)

