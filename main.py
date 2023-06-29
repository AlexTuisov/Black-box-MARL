from pettingzoo.butterfly import knights_archers_zombies_v10
from agent import DQNAgent
from config import *
import numpy as np
import logging
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
        num_archers=4,
        num_knights=0,
        max_zombies=10,
        max_arrows=10,
        pad_observation=True,
        render_mode="human",
        )
    env.reset()
    return env

def main():
    # Create an instance of the environment to get the list of agents
    logging.basicConfig(level=logging.INFO)

    env = define_environment()

    num_episodes = NUM_EPISODES
    agents = {agent: DQNAgent(state_size=np.prod(env.observation_spaces[agent].shape), action_size=env.action_spaces[agent].n) for agent in env.agents}
    agent = env.agents[0]
    total_reward = {agent: 0 for agent in agents.keys()}
    last_reward = {agent: 0 for agent in agents.keys()}
    last_observations = {agent: None for agent in agents.keys()}
    last_actions = {agent: None for agent in agents.keys()}


    for episode in range(num_episodes):
        env.reset()
        print(f"entering episode {episode}")
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            observation = observation.flatten()
            if env.terminations[agent]:
                agents[agent].remember(last_observations[agent], last_actions[agent], DEATH_PENALTY, observation, termination)
                env.step(None)
                continue

            action = agents[agent].act(observation)

            # Normalize reward if necessary
            if last_observations[agent] is not None and last_actions[agent] is not None:
                agents[agent].remember(last_observations[agent], last_actions[agent], reward, observation, termination)
            last_observations[agent] = observation
            last_actions[agent] = action
            total_reward[agent] += reward
            last_reward[agent] += reward

            agents[agent].replay()  # Train the agent using replay
            env.step(action)

            # Update the target network every 'update_target_freq' steps
            if episode % UPDATE_TARGET_FREQ == 0:
                agents[agent].update_target_model()

        if (episode + 1) % LOG_FREQUENCY == 0:
            logging.info(f"last reward: {last_reward}, epsilon: {agents[agent].epsilon}")
            last_reward = {agent: 0 for agent in agents.keys()}

        if WANDB:
            wandb.log({"loss": agents[agent].loss, "average_score": last_reward[agent]})
    if WANDB:
        wandb.finish()


def main_2():
    env = knights_archers_zombies_v10.env(render_mode="human")
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        action = env.action_space(agent).sample() # this is where you would insert your policy
        if env.terminations[agent]:
            env.step(None)
            continue
        env.step(action)

if __name__ == '__main__':
    main()
