import argparse
import pickle
import imageio
from stable_baselines3 import PPO, DQN
from wrappers import *
from collections import namedtuple
from utils import *
import os


def record_gifs():
    env = define_environment_for_playing()
    model = PPO.load(SINGLE_KNIGHT_FILE_PATH)
    for i in range(DATASET_SIZE):
        print(f'starting on gif number {i + 1}')
        images = []
        env.reset()
        j = 0
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            interval_of_responsibility = 1 / KNIGHTS
            agent_index = env.agents.index(agent) % KNIGHTS
            lower_bound, higher_bound = agent_index * interval_of_responsibility, (agent_index + 1) * (
                interval_of_responsibility) + 0.1
            if not np.any(obs) and truncation:
                act = None
            else:
                act = \
                model.predict(transform_array_to_single_knight(obs, lower_bound, higher_bound), deterministic=False)[0]
            env.step(act)
            if j % (2 * len(env.possible_agents)) == 0:
                img = env.render()
                images.append(img)
            j += 1
        # print(images)
        imageio.mimsave(f'gifs/kaz__single_agent_{COORDINATION}_{i}.gif', [np.array(img) for img in images],
                        duration=10)


def load_single_agent_policy():
    env = define_environment_for_playing()
    knight_model = PPO.load(SINGLE_KNIGHT_FILE_PATH)
    archer_model = PPO.load(SINGLE_ARCHER_FILE_PATH)
    for _ in range(5):
        env.reset()
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            interval_of_responsibility = 1 / len(env.agents)
            agent_index = env.agents.index(agent)
            lower_bound, higher_bound = agent_index * interval_of_responsibility - MARGIN, (agent_index + 1) * (
                interval_of_responsibility) + MARGIN
            if not np.any(obs) and truncation:
                act = None
            elif "knight" in agent:
                act = knight_model.predict(transform_array_to_single_knight(obs, lower_bound, higher_bound),
                                           deterministic=False)[0]
            elif "archer" in agent:
                act = archer_model.predict(transform_array_to_single_archer(obs, 0.3),
                                           deterministic=False)[0]
            env.step(act)


def create_dataset():
    env = define_base_environment(visual=False)
    knight_model = PPO.load(SINGLE_KNIGHT_FILE_PATH)
    archer_model = PPO.load(SINGLE_ARCHER_FILE_PATH)
    an_observation = namedtuple('Observation', ['agent',
                                                'observation',
                                                'state',
                                                'action',
                                                'values',
                                                'action_probs',
                                                'reward'])
    for i in range(DATASET_SIZE):
        env.reset()
        trajectory = {agent: [] for agent in env.agents}
        last_obs = {agent: None for agent in env.agents}
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            interval_of_responsibility = 1 / len(env.agents)
            agent_index = env.agents.index(agent)
            lower_bound, higher_bound = agent_index * interval_of_responsibility - MARGIN, (agent_index + 1) * (
                interval_of_responsibility) + MARGIN
            if not np.any(obs) and truncation:
                act = None
            elif "knight" in agent:
                act = knight_model.predict(transform_array_to_single_knight(obs, lower_bound, higher_bound),
                                           deterministic=False)[0]
                try:
                    experience = an_observation(agent,
                                                last_obs[agent],
                                                act,
                                                env.state(),
                                                knight_model.policy.predict_values(last_obs[agent]),
                                                knight_model.policy.predict(last_obs[agent]),
                                                reward)
                except AttributeError:
                    pass
            elif "archer" in agent:
                act = archer_model.predict(transform_array_to_single_archer(obs, 0.3),
                                           deterministic=False)[0]
                try:
                    experience = an_observation(agent,
                                                last_obs[agent],
                                                act,
                                                env.state(),
                                                archer_model.policy.predict_values(last_obs[agent]),
                                                archer_model.policy.predict(last_obs[agent]),
                                                reward)
                except AttributeError:
                    pass
            last_obs[agent] = obs
            try:
                trajectory[agent].append(experience)
            except UnboundLocalError:
                pass
            env.step(act)

        with open(f'data/trajectory_{ARCHERS}_{KNIGHTS}_{SPAWN_RATE}_{i}.pickle', 'wb') as handle:
            pickle.dump(trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_maddpg():
    env_dir = os.path.join('./results', 'kaz')
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    env = define_environment_for_training()
    env.reset()
    dim_info = {}
    for agent in env.possible_agents:
        dim_info[agent] = []
        dim_info[agent].append(env.observation_space(agent).shape[0])
        dim_info[agent].append(env.action_space(agent).n)
    maddpg = MADDPG(dim_info, 100000, 64, 0.001, 0.001, result_dir)

    step = 0  # global step counter
    agent_num = env.num_agents
    episode_num = 100
    random_steps = 100000
    learn_interval = 10000
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(episode_num) for agent_id in env.agents}
    for episode in range(episode_num):
        env.reset()
        obs, reward, termination, truncation, info = env.last()
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        while env.agents:  # interact with the env for an episode
            step += 1
            action = maddpg.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            maddpg.add(obs, action, reward, next_obs, done)

            for agent_id, r in reward.items():  # update reward
                agent_reward[agent_id] += r

            if step >= random_steps and step % learn_interval == 0:  # learn every few steps
                maddpg.learn(64, 0.95)
                maddpg.update_target(0.02)

            obs = next_obs

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        if (episode + 1) % 100 == 0:  # print info every 100 episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward}'
            print(message)

    maddpg.save(episode_rewards)  # save model


def main():
    if TRAIN:
        train()
        load_single_agent_policy()
    elif CREATE_DATASET:
        # record_gifs()
        create_dataset()
    elif COORDINATION:
        load_single_agent_policy()
    else:
        load_policy()


def get_args():
    parser = argparse.ArgumentParser(description="Configuration for the model")

    parser.add_argument('--ppo', type=bool, default=True, help='Use PPO model')
    parser.add_argument('--frames_to_learn', type=int, default=10000000, help='Number of frames to learn')
    parser.add_argument('--spawn_rate', type=int, default=40, help='Spawn rate')
    parser.add_argument('--archers', type=int, default=0, help='Number of archers')
    parser.add_argument('--knights', type=int, default=3, help='Number of knights')
    parser.add_argument('--zombies', type=int, default=3, help='Number of zombies')
    parser.add_argument('--margin', type=float, default=0.05, help='Margin value')
    parser.add_argument('--reward_shaping_factor', type=float, default=0.001, help='Reward shaping factor')
    parser.add_argument('--vector_input', type=bool, default=True, help='Use vector input')
    parser.add_argument('--monitor', type=bool, default=False, help='Monitor the process')
    parser.add_argument('--number_of_games_to_check_model', type=int, default=50, help='Number of games to check model')

    args = parser.parse_args()

    args.model = 'PPO' if args.ppo else 'DQN'
    args.mode = "vectorized" if args.vector_input else "pixelized"
    args.file_path = f'{args.model}_{args.mode}_{args.archers}_{args.knights}_{args.spawn_rate}'
    args.single_knight_file_path = f'{args.model}_{args.mode}_0_1_90'
    args.single_archer_file_path = f'{args.model}_{args.mode}_1_0_40'

    return args


if __name__ == '__main__':
    main()
