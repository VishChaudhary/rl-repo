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
from relaqs.api.callbacks import GateSynthesisCallbacks
import os
from datetime import datetime
from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj, tensor
from qutip.operators import *
import copy
from relaqs.api.utils import *

# path_to_relaqs_root = '/Users/vishchaudhary/rl-repo/src/relaqs'
# QUANTUM_NOISE_DATA_DIR = path_to_relaqs_root + "/quantum_noise_data/"
noise_file = "april/ibmq_belem_month_is_4.json"


def run(train_gate, inference_gate, n_training_iterations=1, n_episodes_for_inferencing = 1, save=True, plot=True, noise_file=noise_file, plot_target_change = True):

    ray.init(num_cpus=7,   # change to your available number of CPUs
        num_gpus=10,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False)

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
    total_Haar_nums = env_config["steps_per_Haar"] * env_config["num_Haar_basis"]

    #************************************************************************************************************************************************#
    ###Check what the need for np.reciprocal is bc the default is not like that
    # env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(),
    #                                        np.reciprocal(t2_list).tolist()]  # using real T1 data
    env_config["relaxation_rates_list"] = [t1_list, t2_list]  # using real T1 data
    #************************************************************************************************************************************************#

    env_config["detuning_list"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(), sigmaz()]
    # env_config["fidelity_threshold"] = 0.7
    # env_config["fidelity_target_switch_case"] = 30
    # env_config["base_target_switch_case"] = 4000
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
    alg_config.num_steps_sampled_before_learning_starts = 10000

    # ---------------------> Tuned Parameters <-------------------------
    alg_config.actor_lr = 5.057359278283752e-05
    alg_config.critic_lr = 9.959658940947128e-05
    alg_config.actor_hiddens = [200] * 10
    alg_config.critic_hiddens = [100] * 10

    # ---------------------> Slightly Higher Fidelity/Reward <-------------------------
    alg_config.exploration_config["random_timesteps"] = 3055.8304716435505
    alg_config.exploration_config["ou_base_scale"] = 0.33536897625927453
    alg_config.exploration_config["ou_theta"] = 0.31360827370009975
    alg_config.exploration_config["ou_sigma"] = 0.26940347674578985
    alg_config.exploration_config["initial_scale"] = 1.469323660064391
    alg_config.exploration_config["scale_timesteps"] = 19885.54898737561
    # alg_config.exploration_config["scale_timesteps"] = 18750
    # alg_config.exploration_config["scale_timesteps"] = int((n_training_iterations * 1000) / (total_Haar_nums * 4))
    # alg_config.exploration_config["scale_timesteps"] = n_training_iterations * 2000

    # ---------------------> Close  Second Fidelity/Reward Parameters <-------------------------
    # alg_config.exploration_config["random_timesteps"] = 4673.765975569726
    # alg_config.exploration_config["ou_base_scale"] = 0.1367000631272562
    # alg_config.exploration_config["ou_theta"] = 0.13792729799506298
    # alg_config.exploration_config["ou_sigma"] = 0.11462192187335964
    # alg_config.exploration_config["initial_scale"] = 0.5497884198832642
    # alg_config.exploration_config["scale_timesteps"] = 10123.97829415627


    # print(f'\nalg_config:\n{alg_config}\n')
    alg = alg_config.build()
    # ---------------------------------------------------------------------

    training_start_time = get_time()
    # ---------------------> Train Agent <-------------------------
    results = [alg.train() for _ in range(n_training_iterations)]
    # -------------------------------------------------------------
    training_end_time = get_time()

    train_env = alg.workers.local_worker().env
    train_alg = alg
    # num_steps_done = train_env.get_self_episode_num()
#2024-12-30_11-53-34
    # original_episodes_target_switch = train_env.get_episodes_gate_switch()
    # gate_switch_array = None
    #
    # if plot_target_change:
    #     episodes_target_switch = original_episodes_target_switch.copy()
    #     for idx in range(len(episodes_target_switch)):
    #         episode = episodes_target_switch[idx]
    #         episode = int((episode - env_config["steps_per_Haar"]) / total_Haar_nums)
    #         episodes_target_switch[idx] = episode
    #
    #     gate_switch_array = episodes_target_switch

    save_dir = None

    # ---------------------> Save/Plot Training Results <-------------------------
    if save and plot is True:
        sr = SaveResults(train_env, alg, save_path=save_filepath,
                         target_gate_string=f"Noisy_Train-{str(train_gate)}, Inference-{str(inference_gate)}")
        save_dir = sr.save_results()
        plot_data(save_dir, plot_filename=training_plot_filename,
                  figure_title=f"[NOISY] Training on {str(train_gate)}", gate_switch_array=None)
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    config_table(env_config, alg_config, save_filepath, plot_target_change, n_training_iterations)

    for inferencing_gate in inference_gate:
        # train_alg = copy.deepcopy(alg)
        inferencing_plot_filename = f'inferencing_{inferencing_gate}.png'
        # -----------------------> Inferencing <---------------------------
        env = train_alg.workers.local_worker().env
        inference_env, inf_alg, calculated_target, target_gate = do_inferencing(env, train_alg, inferencing_gate, n_episodes_for_inferencing)

        # ---------------------> Save/Plot Inference Results <-------------------------
        if plot is True:
            plot_data(save_dir, plot_filename=inferencing_plot_filename, env=inference_env, inference=True,
                      figure_title=f"[NOISY] Inferencing on {str(inferencing_gate)} (Previously Trained on {str(train_gate)})")
            create_self_U_textfile(save_filepath, inferencing_gate, calculated_target, target_gate)

        # --------------------------------------------------------------



    # if plot_target_change:
    #     print(f'Original Switching Array: {original_episodes_target_switch}')
    #     print(f'Adjusted Switching Array: {episodes_target_switch}')
    # else:
    #     print(f'Not plotting when target switches. Number of times target switches: {len(original_episodes_target_switch)}')

    training_elapsed_time = training_end_time - training_start_time
    print(f"Training Elapsed time: {training_elapsed_time}\n")

    ray.shutdown()


def do_inferencing(env, alg, inferencing_gate, n_episodes_for_inferencing):
    """
    alg: The trained model
    n_episodes_for_inferencing: Number of episodes to do during the training
    """

    assert n_episodes_for_inferencing > 0

    # Initialize a new environment for inference using this configuration
    inference_env_config = env.return_env_config()
    target_gate = inferencing_gate.get_matrix()  # Set new target gate for inference
    inference_env_config["U_target"] = target_gate
    inference_env_config['training'] = False
    inference_env_config['verbose'] = False
    inference_env = NoisySingleQubitEnv(inference_env_config)

    # ------------------------------------------------------------------------------------
    target_gate = np.array(target_gate)

    super_op_target_gate = (spre(Qobj(target_gate)) * spost(Qobj(target_gate))).data.toarray()
    U_target_dagger = np.array(super_op_target_gate.conjugate().transpose())
    # ------------------------------------------------------------------------------------

    num_episodes = 0
    episode_reward = 0.0
    print(f'Inference Gate Name: {inferencing_gate}\n')
    print("*************************************************************************************************")

    obs, info = inference_env.reset()  # Start with the inference environment
    calculated_target = {}
    while num_episodes < n_episodes_for_inferencing:
        # print("Episode : ", num_episodes)
        # Compute an action (`a`).
        action = alg.compute_single_action(
            observation=obs,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `action` to the env.
        obs, reward, done, truncated, _ = inference_env.step(action)
        episode_reward += reward
        # print("####################################################")
        self_U = np.array(inference_env.get_self_U())
        fidelity = float(np.abs(np.trace(U_target_dagger @ self_U))) / (self_U.shape[0])
        calculated_target[fidelity] = self_U
        # print(f'Episode: {num_episodes}\nSelf.U:\n{calculated_gate}')
        # Is the episode `done`? -> Reset.
        if done:
            # print(f"Episode done: Total reward = {episode_reward}")
            obs, info = inference_env.reset()
            num_episodes += 1
            episode_reward = 0.0

    return inference_env, alg, calculated_target, target_gate


if __name__ == "__main__":
    n_training_iterations = 1
    n_episodes_for_inferencing = 20

    save = True
    plot = True
    plot_target_change = False

    train_gate = gates.RandomSU2()

    inferencing_gate = [gates.RandomSU2(), gates.I(), gates.X(), gates.Y(), gates.Z(), gates.H(), gates.S(), gates.X_pi_4()]
    run(train_gate, inferencing_gate, n_training_iterations, n_episodes_for_inferencing, save, plot, noise_file, plot_target_change)
