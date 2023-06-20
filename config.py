BUFFER_SIZE = 2000  # max size of the replay buffer
BATCH_SIZE = 32  # number of experiences to sample from the buffer
GAMMA = 0.95  # discount factor
EPSILON = 1.0  # initial exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99
LEARNING_RATE = 0.001