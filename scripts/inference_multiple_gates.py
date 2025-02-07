""" Learning new single qubit gates, rather than the default X gate. """

import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.noisy_single_qubit_env import NoisySingleQubitEnv
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
import relaqs.api.gates as gates
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
from relaqs.api.callbacks import GateSynthesisCallbacks
import os
from datetime import datetime
from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj, tensor
from qutip.operators import *
import copy
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import qutip
import torch
from relaqs.api.utils import *

# path_to_relaqs_root = '/Users/vishchaudhary/rl-repo/src/relaqs'
# QUANTUM_NOISE_DATA_DIR = path_to_relaqs_root + "/quantum_noise_data/"
noise_file = "april/ibmq_belem_month_is_4.json"


def run(train_gate, inference_gate, n_training_iterations=1, n_episodes_for_inferencing = 1, save=True, plot=True, noise_file=noise_file, plot_target_change = True):

# 14 CPU Cores
# 20 GPU Cores

    ray.init(num_cpus=14,   # change to your available number of CPUs
        num_gpus=20,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False)
    # print(ray.available_resources())

    # ---------------------> Configure algorithm and Environment <-------------------------
    # Initialize default configuration
    env_config = NoisySingleQubitEnv.get_default_env_config()

    save_filepath = "/Users/vishchaudhary/rl-repo/results/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
    training_plot_filename = f'training_{train_gate}.png'


    # ---------------------> Get quantum noise data <-------------------------
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)

    # ---------------------> Configure Environment <-------------------------
    env_config["U_target"] = train_gate.get_matrix()
    env_config['num_Haar_basis'] = 1
    env_config['steps_per_Haar'] = 2
    env_config['training'] = True
    env_config['retraining'] = False

    #************************************************************************************************************************************************#
    ###Check what the need for np.reciprocal is bc the default is not like that
    # env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(),
    #                                        np.reciprocal(t2_list).tolist()]  # using real T1 data
    env_config["relaxation_rates_list"] = [t1_list, t2_list]  # using real T1 data
    #************************************************************************************************************************************************#

    env_config["detuning_list"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(), sigmaz()]
    env_config["verbose"] = False

    # ---------------------> Configure algorithm<-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    alg_config.environment(NoisySingleQubitEnv, env_config=env_config)
    alg_config.callbacks(GateSynthesisCallbacks)
    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.train_batch_size = env_config["steps_per_Haar"]

    ### working 1-3 sets
    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    # alg_config.num_steps_sampled_before_learning_starts = 10000

    # # ---------------------> Tuned Parameters <-------------------------
    alg_config.actor_lr = 5.057359278283752e-05
    alg_config.critic_lr = 9.959658940947128e-05
    alg_config.actor_hiddens = [200] * 10
    alg_config.critic_hiddens = [100] * 10

    # alg_config.actor_hiddens = [256] * 10
    # alg_config.critic_hiddens = [200] * 10

    # ---------------------> 2025-01-22_05-50-08-HPT: Tuned Parameters <-------------------------
    # alg_config.actor_lr = 2.5512592152219747e-05
    # alg_config.critic_lr = 1.3097861849113528e-05
    # alg_config.actor_hiddens = [100] * 50
    # alg_config.critic_hiddens = [200] * 75
    # alg_config.num_steps_sampled_before_learning_starts = 5000

    # # ---------------------> 2025-01-23_05-14-17-HPT: Tuned Parameters <-------------------------
    # alg_config.actor_lr = 1.822167059368299e-05
    # alg_config.critic_lr = 4.500308062559224e-05
    # alg_config.actor_hiddens = [50] * 50
    # alg_config.critic_hiddens = [300] * 30
    # alg_config.num_steps_sampled_before_learning_starts = 10000

    # ---------------------> Slightly Higher Fidelity/Reward <-------------------------
    alg_config.exploration_config["random_timesteps"] = 3055.8304716435505
    alg_config.exploration_config["ou_base_scale"] = 0.33536897625927453
    alg_config.exploration_config["ou_theta"] = 0.31360827370009975
    alg_config.exploration_config["ou_sigma"] = 0.26940347674578985
    # alg_config.exploration_config["initial_scale"] = 1.469323660064391
    alg_config.exploration_config["initial_scale"] = 1.1
    # alg_config.exploration_config["initial_scale"] = 1.0

    # alg_config.exploration_config["scale_timesteps"] = 20000
    # alg_config.exploration_config["scale_timesteps"] = 18750
    alg_config.twin_q = True
    # alg_config.smooth_target_policy = True


    # print(f'\nalg_config:\n{alg_config}\n')
    # alg_config = alg_config.resources(num_gpus=1)
    alg = alg_config.build()
    # alg = alg.to(torch.device("mps"))
    # ---------------------------------------------------------------------

    training_start_time = get_time()
    # print(ray.available_resources())
    # ---------------------> Train Agent <-------------------------
    n_training_iterations *= env_config['num_Haar_basis'] * env_config['steps_per_Haar']

    update_every_percent = 5
    results = []
    # update_interval = n_training_iterations * (update_every_percent / 100)
    update_interval =  max(1, int(n_training_iterations * (update_every_percent / 100)))

    for i in range(n_training_iterations):
        results.append(alg.train())
        # Print update every x%
        if (i + 1) % int(update_interval) == 0 or (i + 1) == n_training_iterations:
            percent_complete = (i + 1) / n_training_iterations * 100
            print(f"Training Progress: {percent_complete:.0f}% complete")

    # results = [alg.train() for _ in range(n_training_iterations)]
    # -------------------------------------------------------------
    training_end_time = get_time()

    train_env = alg.workers.local_worker().env
    train_alg = alg

    save_dir = None

    # ---------------------> Save/Plot Training Results <-------------------------
    if save and plot is True:
        sr = SaveResults(train_env, alg, save_path=save_filepath)
        save_dir = sr.save_results()
        plot_data(save_dir, plot_filename=training_plot_filename,
                  figure_title=f"[NOISY] Training on {str(train_gate)}", gate_switch_array=None)
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    config_table(env_config = env_config, alg_config = alg_config, filepath = save_filepath)
    columns = ['Fidelity', 'Rewards', 'Actions', 'Self.U_Operator', 'U_target', 'Episode Id']
    inf_count = 0

    for gate in inference_gate:
        # train_alg = copy.deepcopy(alg)

        gate_save_dir = save_dir+f'/{gate}/'
        plot_filename = f'inference_{gate}.png'
        os.makedirs(gate_save_dir)

        if inf_count == 0:
            figure_title = f"[NOISY] Inferencing on Multiple Different {str(gate)}. (Previously Trained on Multiple {str(train_gate)})"
        else:
            figure_title = f"[NOISY] Multiple Instances of Inferencing on {str(gate)} (Previously Trained on Multiple {str(train_gate)})"
        env_data_title = f"{gate}_"
        transition_history = []

        for inference_iteration in range(n_episodes_for_inferencing):
            # -----------------------> Inferencing <---------------------------
            env = train_alg.workers.local_worker().env
            inference_env, target_gate, history = do_inferencing(env, train_alg, gate)
            transition_history.append(history)

        df = pd.DataFrame(transition_history, columns=columns)
        # df.to_pickle(env_data_title + "env_data.pkl")  # easier to load than csv
        df.to_csv(gate_save_dir + env_data_title + "env_data.csv", index=False)  # backup in case pickle doesn't work
        plot_multiple_visuals(df, figure_title=figure_title, save_dir= gate_save_dir, plot_filename=plot_filename, inf_count = inf_count, gate=gate)

        inf_count += 1

        # ---------------------> Save/Plot Inference Results <-------------------------
        # if plot is True:
        #     plot_data(env_data_title, plot_filename=inferencing_plot_filename,
        #               figure_title=figure_title, modified_inference=True, df = df)
        # --------------------------------------------------------------



    # if plot_target_change:
    #     print(f'Original Switching Array: {original_episodes_target_switch}')
    #     print(f'Adjusted Switching Array: {episodes_target_switch}')
    # else:
    #     print(f'Not plotting when target switches. Number of times target switches: {len(original_episodes_target_switch)}')

    training_elapsed_time = training_end_time - training_start_time
    print(f"Training Elapsed time: {training_elapsed_time}\n")
    print(f'File saved to: {save_filepath}')

    ray.shutdown()




def do_inferencing(env, alg, gate):
    """
    alg: The trained model
    n_episodes_for_inferencing: Number of episodes to do during the training
    """

    # Initialize a new environment for inference using this configuration
    inference_env_config = env.return_env_config()
    target_gate = gate.get_matrix()  # Set new target gate for inference

    inference_env_config["U_target"] = target_gate
    inference_env_config['training'] = False
    inference_env_config['verbose'] = False
    inference_env_config['retraining'] = False
    inference_env = NoisySingleQubitEnv(inference_env_config)

    # ------------------------------------------------------------------------------------
    target_gate = np.array(target_gate)

    episode_reward = 0.0
    done = False

##Check this out. When we reset it computes the fidelity for the observation between the U_inital (I gate by default) and our new target for the first haar num and step per haar
    obs, info = inference_env.reset()  # Start with the inference environment
    while not done:

        # Compute an action (`a`).
        action = alg.compute_single_action(
            observation=obs,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `action` to the env.
        obs, reward, done, truncated, _ = inference_env.step(action)
        episode_reward += reward

        if done:
            return inference_env, target_gate, inference_env.transition_history[-1]


def main():
    # Modified to be number of episodes for training (in thousands)
    n_training_iterations = 75
    n_episodes_for_inferencing = 1000

    save = True
    plot = True
    plot_target_change = False

    train_gate = gates.RandomSU2()

    ##RandomGate must be kept as first in the array and XY_combination MUST be kept as second in the array
    inferencing_gate = [gates.RandomSU2(), gates.XY_combination(), gates.I(), gates.X(), gates.Y(), gates.Z(), gates.H(), gates.S(),
                        gates.X_pi_4()]

    run(train_gate, inferencing_gate, n_training_iterations, n_episodes_for_inferencing, save, plot, noise_file,
        plot_target_change)

if __name__ == "__main__":
    main()