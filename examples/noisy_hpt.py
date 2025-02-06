""" Example of hyperparameter search over DDPG exploration config"""
import os
import ray
from ray import tune
from ray.air import RunConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from relaqs.environments import NoisySingleQubitEnv
from ray.tune.search.optuna import OptunaSearch
from relaqs import RESULTS_DIR
from datetime import datetime
import numpy as np
import relaqs.api.gates as gates
from relaqs.api.callbacks import GateSynthesisCallbacks
from relaqs.api.utils import *
from ray.train import ScalingConfig

path_to_relaqs_root = '/Users/vishchaudhary/rl-repo/src/relaqs'
QUANTUM_NOISE_DATA_DIR = path_to_relaqs_root + "/quantum_noise_data/"
noise_file = "april/ibmq_belem_month_is_4.json"

def save_hpt_table(results: tune.ResultGrid):
    df = results.get_dataframe()
    path = RESULTS_DIR + datetime.now().strftime("%Y-%m-%d_%H-%M-%S-HPT/")
    os.makedirs(path)
    df.to_csv(path + "hpt_results.csv")

def run_ray_tune(environment, n_configurations=100, n_training_iterations=50, save=True):

    ray.init(num_cpus=14,  # change to your available number of CPUs
             num_gpus=20,
             include_dashboard=False,
             ignore_reinit_error=True,
             log_to_driver=False)

    search_space = {"environment" : environment,
                    "n_training_iterations" : n_training_iterations,
                    "random_timesteps": tune.uniform(100, 10000),
                    "ou_base_scale": tune.uniform(0.01, 0.5),
                    "ou_theta": tune.uniform(0.05, 0.5),
                    "ou_sigma": tune.uniform(0.05, 0.5),
                    "initial_scale": tune.uniform(0.1, 2.0),
                    "scale_timesteps": tune.uniform(1000, 100000),
                    "num_gpus": tune.choice([1, 2, 5]),  # Let Tune allocate GPU cores dynamically
                    "num_cpus": tune.choice([2, 4, 6]),  # Let Tune allocate CPU cores dynamically
                    }

    # scaling_config = ScalingConfig(
    #     num_workers=4,  # Number of parallel workers per trial
    #     use_gpu=True,  # Enable GPU usage
    #     resources_per_worker={"CPU": 2, "GPU": 5},  # Allocate resources per worker
    # )
    algo = OptunaSearch()
    tuner = tune.Tuner(
        objective,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="avg_final_fidelities",
            mode="max",
            search_alg=algo,
            num_samples=n_configurations,
            # resources_per_trial={"cpu": 4, "gpu": 1}  # Allocate 4 CPUs and 1 GPU per trial
            ),
        run_config=RunConfig(
            stop={"training_iteration": n_training_iterations},
            # scaling_config=scaling_config  # Apply the scaling config
        ),
        )
    results = tuner.fit()
    best_fidelity_config = results.get_best_result(metric="avg_final_fidelities", mode="max").config
    print("best_fidelity_config", best_fidelity_config)

    if save is True:
        save_hpt_table(results)

def objective(config):
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)
    train_gate = gates.RandomSU2()
    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    env_config = config["environment"].get_default_env_config()
    env_config["U_target"] = train_gate.get_matrix()
    env_config["detuning_list"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(), sigmaz()]
    env_config["training"] = True
    env_config['num_Haar_basis'] = 1
    env_config['steps_per_Haar'] = 2
    env_config["verbose"] = False
    alg_config.environment(config["environment"], env_config=env_config)
    alg_config.callbacks(GateSynthesisCallbacks)
    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.train_batch_size = env_config["steps_per_Haar"]
    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    alg_config.num_steps_sampled_before_learning_starts = 5000

    # ---------------------> Tuned Parameters <-------------------------
    alg_config.actor_lr = 5.057359278283752e-05
    alg_config.critic_lr = 9.959658940947128e-05
    alg_config.actor_hiddens = [200] * 10
    alg_config.critic_hiddens = [100] * 10
    alg_config.twin_q = True

    exploration_config_keys = alg_config["exploration_config"].keys()
    for key, value in config.items():
        if key in exploration_config_keys:
            alg_config.exploration_config[key] = value

    alg = alg_config.build()
    # ---------------------------------------------------------------------

    # Train
    results = [alg.train() for _ in range(config["n_training_iterations"])]

    # Record
    env = alg.workers.local_worker().env
    fidelities = [transition[0] for transition in env.transition_history]
    averageing_window = 50 if len(fidelities) >= 50 else len(fidelities)
    avg_final_fidelities = np.mean([fidelities[-averageing_window:]])
    results = {
            "max_fidelity": max(fidelities),
            "avg_final_fidelities" : avg_final_fidelities,
            "final_fidelity" : fidelities[-1],
            "final_reward" : env.transition_history[-1][1]
        }
    return results

if __name__ == "__main__":
    environment = NoisySingleQubitEnv
    n_configurations = 28
    n_training_iterations = 150
    save = True
    run_ray_tune(environment, n_configurations, n_training_iterations, save)
    ray.shutdown() # not sure if this is required
