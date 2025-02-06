import os
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
import torch


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
            "actor_lr" : tune.loguniform(1e-6,1e-4),
            "critic_lr" : tune.loguniform(1e-6,1e-4),
            "actor_num_hiddens" : tune.choice([30, 50, 75, 100, 150]),
            "actor_layer_size" : tune.choice([30, 50, 100, 200, 300]),
            "critic_num_hiddens" : tune.choice([30, 50, 75, 100, 150]),
            "critic_layer_size" : tune.choice([30, 50, 100, 200, 300])
            }

    algo = OptunaSearch()
    tuner = tune.Tuner(
        objective,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="avg_final_fidelities",
            mode="max",
            search_alg=algo,
            num_samples=n_configurations
            ),
            run_config = RunConfig(
                stop={"training_iteration": n_training_iterations},
            ),
        )
    results = tuner.fit()
    best_fidelity_config = results.get_best_result(metric="avg_final_fidelities", mode="max").config
    print("best_fidelity_config", best_fidelity_config)
    
    # Average within scope
    #best_avg_fidelity_config = results.get_best_result(metric="fidelity", mode="max", scope="last-50-avg").config
    #print("best_avg_fidelity_config", best_avg_fidelity_config)
    
    if save is True:
        save_hpt_table(results)

def objective(config):
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)
    train_gate = gates.RandomSU2()
    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    # alg_config.resources(num_gpus=1)
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
    alg_config.num_steps_sampled_before_learning_starts = 10000
    alg_config.exploration_config["random_timesteps"] = 3055.8304716435505
    alg_config.exploration_config["ou_base_scale"] = 0.33536897625927453
    alg_config.exploration_config["ou_theta"] = 0.31360827370009975
    alg_config.exploration_config["ou_sigma"] = 0.26940347674578985
    alg_config.exploration_config["initial_scale"] = 1.469323660064391
    alg_config.exploration_config["scale_timesteps"] = 18750

    alg_config.twin_q = True

    alg_config.actor_lr = config["actor_lr"]
    alg_config.critic_lr = config["critic_lr"]

    alg_config.actor_hiddens = [config["actor_layer_size"]] * int(config["actor_num_hiddens"])
    alg_config.critic_hiddens = [config["critic_layer_size"]] * int(config["critic_num_hiddens"])

    alg = alg_config.build()
    # device = torch.device("mps")
    #
    # for policy_id, policy in alg.workers.local_worker().policy_map.items():
    #     policy.model.to(device)
    #
    #     # Move the optimizer's state to MPS as well (important for PyTorch)
    #     for state in policy.model.state_dict().values():
    #         if isinstance(state, torch.Tensor):
    #             state.data = state.data.to(device)
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
    ray.shutdown()  # not sure if this is required
    