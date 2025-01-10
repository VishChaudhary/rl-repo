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

########################################################################################################################
#               POTENTIALLY OUTDATED- DO NOT USE WITHOUT COMPARING TO OTHER FILES FIRST
########################################################################################################################

def continue_training(train_gate="RandomSU2", inference_gate= None, n_training_iterations=1, n_episodes_for_inferencing = 1, save=True, plot=True, plot_target_change = False, original_training_date = None):
    save_filepath = "/Users/vishchaudhary/rl-repo/results/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
    training_plot_filename = f'training_{train_gate}.png'

    alg = load_model(base_path)
    train_env = alg.workers.local_worker().env
    alg_config = alg.workers.local_worker().config
    env_config = train_env.return_env_config()

    total_Haar_nums = env_config["steps_per_Haar"] * env_config["num_Haar_basis"]

    training_start_time = get_time()
    # ---------------------> Train Agent <-------------------------
    results = [alg.train() for _ in range(n_training_iterations)]
    # -------------------------------------------------------------
    training_end_time = get_time()

    train_env = alg.workers.local_worker().env
    train_alg = alg

    num_steps_done = train_env.get_self_episode_num()
    original_episodes_target_switch = train_env.get_episodes_gate_switch()
    gate_switch_array = None

    if plot_target_change:
        episodes_target_switch = original_episodes_target_switch.copy()
        for idx in range(len(episodes_target_switch)):
            episode = episodes_target_switch[idx]
            episode = int((episode - env_config["steps_per_Haar"]) / total_Haar_nums)
            episodes_target_switch[idx] = episode

        gate_switch_array = episodes_target_switch

    save_dir = None

    # ---------------------> Save/Plot Training Results <-------------------------
    if save and plot is True:
        sr = SaveResults(train_env, alg, save_path=save_filepath,
                         target_gate_string=f"Noisy_Train-{str(train_gate)}, Inference-{str(inference_gate)}")
        save_dir = sr.save_results()
        plot_data(save_dir, plot_filename=training_plot_filename,
                  episode_length=alg._episode_history[0].episode_length,
                  figure_title=f"[NOISY] Training on {str(train_gate)}", gate_switch_array=gate_switch_array)
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    config_table(env_config, alg_config, save_filepath, plot_target_change, n_training_iterations, continue_training= True, original_training_date= original_training_date)

    for inferencing_gate in inference_gate:
        # train_alg = copy.deepcopy(alg)
        inferencing_plot_filename = f'inferencing_{inferencing_gate}.png'
        # -----------------------> Inferencing <---------------------------
        env = train_alg.workers.local_worker().env
        inference_env, inf_alg, final_gate_kron = do_inferencing(env, train_alg, inferencing_gate, n_episodes_for_inferencing)

        # ---------------------> Save/Plot Inference Results <-------------------------
        if plot is True:
            plot_data(save_dir, plot_filename=inferencing_plot_filename, env=inference_env, inference=True,
                      episode_length=inf_alg._episode_history[0].episode_length,
                      figure_title=f"[NOISY] Inferencing on {str(inferencing_gate)} (Previously Trained on {str(train_gate)})")
            create_self_U_textfile(save_filepath, inferencing_gate, final_gate_kron)
            print("Results saved to:", save_dir)
        # --------------------------------------------------------------
    print(f'Num times steps called: {num_steps_done}\nWhile training for training iterations num: {n_training_iterations}\n')
    num_haar_basis = env_config['num_Haar_basis']
    num_steps_haar = env_config['steps_per_Haar']
    print(f'Num Haar basis: {num_haar_basis}\nNum steps per haar: {num_steps_haar}')
    print(f'Type: {type(train_env)}')

    if plot_target_change:
        print(f'Original Switching Array: {original_episodes_target_switch}')
        print(f'Adjusted Switching Array: {episodes_target_switch}')
    else:
        print(f'Not plotting when target switches. Number of times target switches: {len(original_episodes_target_switch)}')

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
    inference_env_config["U_target"] = inferencing_gate.get_matrix()  # Set new target gate for inference
    inference_env_config['switch_every_episode'] = False
    inference_env_config['verbose'] = False
    print(f'U_target:\n{inference_env_config["U_target"]}\n\n')
    inference_env = NoisySingleQubitEnv(inference_env_config)

    num_episodes = 0
    episode_reward = 0.0
    print("Inferencing on a different gate is starting ....")
    print("*************************************************************************************************")

    obs, info = inference_env.reset()  # Start with the inference environment
    final_gate_kron = []
    while num_episodes < n_episodes_for_inferencing:
        print("Episode : ", num_episodes)
        # Compute an action (`a`).
        action = alg.compute_single_action(
            observation=obs,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `action` to the env.
        obs, reward, done, truncated, _ = inference_env.step(action)
        episode_reward += reward
        print("####################################################")
        final_gate_kron = np.array(inference_env.get_self_U())
        print(f'Episode: {num_episodes}\nSelf.U:\n{final_gate_kron}')
        # Is the episode `done`? -> Reset.
        if done:
            print(f"Episode done: Total reward = {episode_reward}")
            obs, info = inference_env.reset()
            num_episodes += 1
            episode_reward = 0.0

    return inference_env, alg, final_gate_kron


if __name__ == "__main__":
    ########################################################################################################################
    #               POTENTIALLY OUTDATED- DO NOT USE WITHOUT COMPARING TO OTHER FILES FIRST
    ########################################################################################################################

    # base_path = "/Users/vishchaudhary/rl-repo/results/2024-12-30_11-53-34/model_checkpoints"
    # base_path = "/Users/vishchaudhary/rl-repo/results/2024-12-30_14-56-24/model_checkpoints"
    base_path = "/Users/vishchaudhary/rl-repo/results/2024-12-31_20-32-49/model_checkpoints"
    original_training_date = "2024-12-31_20-32-49"

    n_training_iterations = 150
    n_episodes_for_inferencing = 100

    save = True
    plot = True
    plot_target_change = False

    inferencing_gate = [gates.I(), gates.X_pi_4(), gates.X(), gates.Z(), gates.H(), gates.S(), gates.RandomSU2(),
                        gates.Y()]
    continue_training(inference_gate=inferencing_gate, n_training_iterations=n_training_iterations,
                      n_episodes_for_inferencing=n_episodes_for_inferencing, save=save, plot=plot,
                      plot_target_change=plot_target_change, original_training_date=original_training_date)
