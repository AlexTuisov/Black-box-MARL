import pickle
import imageio
from stable_baselines3 import PPO, DQN
from wrappers import *
from collections import namedtuple
from utils import *




def record_gifs():
    env = define_environment_for_playing()
    model = PPO.load(SINGLE_KNIGHT_FILE_PATH)
    for i in range(DATASET_SIZE):
        print(f'starting on gif number {i+1}')
        images = []
        env.reset()
        j = 0
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            interval_of_responsibility = 1 / KNIGHTS
            agent_index = env.agents.index(agent) % KNIGHTS
            lower_bound, higher_bound = agent_index*interval_of_responsibility, (agent_index+1)*(interval_of_responsibility) + 0.1
            if not np.any(obs) and truncation:
                act = None
            else:
                act = model.predict(transform_array_to_single_knight(obs, lower_bound, higher_bound), deterministic=False)[0]
            env.step(act)
            if j % (2 * len(env.possible_agents)) == 0:
                img = env.render()
                images.append(img)
            j += 1
        # print(images)
        imageio.mimsave(f'gifs/kaz__single_agent_{COORDINATION}_{i}.gif', [np.array(img) for img in images], duration=10)

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
            lower_bound, higher_bound = agent_index*interval_of_responsibility - MARGIN, (agent_index+1)*(interval_of_responsibility) + MARGIN
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



if __name__ == '__main__':
    main()
