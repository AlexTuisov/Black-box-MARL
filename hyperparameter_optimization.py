import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import gymnasium as gym
import torch
from stable_baselines3.common.callbacks import EvalCallback
from main import define_environment_for_training
from config import *
from stable_baselines3 import PPO
import time


def sample_PPO_params(trial: optuna.Trial):
    """Sampler for A2C hyperparameters."""
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)

    learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    # batch_size = 2 ** trial.suggest_int("exponent_batch_size", 3, 8)
    n_epochs = 2 ** trial.suggest_int("exponent_n_epochs", 2, 6)


    # Display true values.
    trial.set_user_attr("gamma_", gamma)


    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "n_epochs": n_epochs,
    }



def objective(trial: optuna.Trial) -> float:
    # Sample hyperparameters.
    print("starting new sample")
    print(time.time() - MYTIME)
    kwargs = sample_PPO_params(trial)
    kwargs.update({
        "policy": "MlpPolicy",
        "env": define_environment_for_training()
    })
    # Create the RL model.
    model = PPO(**kwargs)
    env = define_environment_for_training()
    # Create env used for evaluation.
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if needed.

            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

def optuna_main():

    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    print("finished warmup")

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=60000)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    optuna_main()
