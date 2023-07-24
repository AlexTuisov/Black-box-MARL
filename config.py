BUFFER_SIZE = 10000  # max size of the replay buffer
BATCH_SIZE = 32  # number of experiences to sample from the buffer
GAMMA = 0.95  # discount factor
EPSILON = 1.0  # initial exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9999
LEARNING_RATE = 0.001
UPDATE_TARGET_FREQ = 1
DEATH_PENALTY = -10
LOG_FREQUENCY = 10
HIDDEN_DQN_SIZE = 32
NUM_EPISODES = 500
MAX_GRAD = 1
SPAWN_RATE = 20
WANDB = False

