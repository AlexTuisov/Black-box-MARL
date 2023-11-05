import time
ppo = True
MODEL = 'PPO' if ppo else 'DQN'
FRAMES_TO_LEARN = 10000000
SPAWN_RATE = 20
ARCHERS = 0
KNIGHTS = 1
ZOMBIES = 1
MARGIN = 0.05
# TRAIN_Y_LOWER_LIMIT, TRAIN_Y_UPPER_LIMIT = 0.3, 1
REWARD_SHAPING_FACTOR = 0.001
VECTOR_INPUT = True
mode = "vectorized" if VECTOR_INPUT else "pixelized"
FILE_PATH = f'{MODEL}_{mode}_{ARCHERS}_{KNIGHTS}_{SPAWN_RATE}'
SINGLE_KNIGHT_FILE_PATH = f'{MODEL}_{mode}_0_1_60'
SINGLE_ARCHER_FILE_PATH = f'{MODEL}_{mode}_1_0_40'
MONITOR = False

#optimized hyperparameters for 2 agents
if ARCHERS + KNIGHTS < 3:
    NUM_EPOCHS = 2**4
    NUM_STEPS = 2**10
    LEARNING_RATE = 0.000493
    GAMMA = 1-0.0135
    ENTROPY_COEFFICIENT = 0.000778

#optimized hyperparameters for 3 agents
else:
    NUM_EPOCHS = 2**3
    NUM_STEPS = 2**10
    LEARNING_RATE = 0.00021
    GAMMA = 1-0.00393
    ENTROPY_COEFFICIENT = 0.0000708


REWARD_SHAPING = False
CREATE_DATASET = False
TRAIN = False
COORDINATION = False
MAX_NO_IMPROVEMENT_EVALS = 5
DATASET_SIZE = 1000

# optuna hyperparams
N_TRIALS = 50
N_STARTUP_TRIALS = N_TRIALS // 10
N_EVALUATIONS = 5
N_EVAL_EPISODES = 10
N_TIMESTEPS = 500000
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
MYTIME = time.time()
