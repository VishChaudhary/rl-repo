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


def continue_training(train_gate="RandomSU2", inference_gate= None, n_episodes_for_inferencing = 1, save=True, plot=True, plot_target_change = False, original_training_date = None):
    save_filepath = "/Users/vishchaudhary/rl-repo/results/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
    training_plot_filename = f'training_{train_gate}.png'

    alg = load_model(base_path)
    train_env = alg.workers.local_worker().env
    alg_config = alg.workers.local_worker().config
    env_config = train_env.return_env_config()

    total_Haar_nums = env_config["steps_per_Haar"] * env_config["num_Haar_basis"]

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
    for inferencing_gate in inference_gate:
        # train_alg = copy.deepcopy(alg)
        inferencing_plot_filename = f'inferencing_{inferencing_gate}.png'
        # -----------------------> Inferencing <---------------------------
        env = train_alg.workers.local_worker().env
        inference_env, inf_alg, calculated_target, target_gate = do_inferencing(env, train_alg, inferencing_gate, n_episodes_for_inferencing)


        # ---------------------> Save/Plot Inference Results <-------------------------
        if plot is True:
            plot_data(save_dir = save_dir, plot_filename=inferencing_plot_filename, env=inference_env, inference=True,
                      figure_title=f"[NOISY] Inferencing on {str(inferencing_gate)} (Previously Trained on {str(train_gate)})")
            # create_self_U_textfile(save_filepath, inferencing_gate, final_gate_kron)
            # print("Results saved to:", save_dir)
            # target_gate = inferencing_gate.get_matrix()
            super_op_target_gate = (spre(Qobj(target_gate)) * spost(Qobj(target_gate))).data.toarray()
            # file_name = save_filepath + f"{inference_gate}_self_U.txt"
            U_target_dagger = np.array(super_op_target_gate.conjugate().transpose())

            # fidelity = float(np.abs(np.trace(U_target_dagger @ final_gate_kron))) / (final_gate_kron.shape[0])
            # print(f'Super_op_target_gate:\n{super_op_target_gate}\n\n')
            # print(f'Final Self.U:\n{final_gate_kron}\n\n')
            # print(f'Fidelity: {fidelity}')
            highest_val_self_U = np.array(calculated_target[max(calculated_target.keys())])
            fidelity = float(np.abs(np.trace(U_target_dagger @ highest_val_self_U))) / (highest_val_self_U.shape[0])
            # print(f'Super_op_target_gate:\n{np.array(super_op_target_gate)}\n\n')
            # print(f'Final Self.U:\n{np.array(highest_val_self_U)}\n\n')
            print(f'Fidelity: {fidelity}')
        # --------------------------------------------------------------
    # num_haar_basis = env_config['num_Haar_basis']
    # num_steps_haar = env_config['steps_per_Haar']
    # print(f'Num Haar basis: {num_haar_basis}\nNum steps per haar: {num_steps_haar}')
    # print(f'Type: {type(train_env)}')
    #
    # if plot_target_change:
    #     print(f'Original Switching Array: {original_episodes_target_switch}')
    #     print(f'Adjusted Switching Array: {episodes_target_switch}')
    # else:
    #     print(f'Not plotting when target switches. Number of times target switches: {len(original_episodes_target_switch)}')



    ray.shutdown()


def do_inferencing(env, alg, inferencing_gate, n_episodes_for_inferencing):
    """
    alg: The trained model
    n_episodes_for_inferencing: Number of episodes to do during the training
    """

    assert n_episodes_for_inferencing > 0

    # Initialize a new environment for inference using this configuration
    inference_env_config = env.return_env_config()
    target_gate = inferencing_gate.get_matrix()
    inference_env_config["U_target"] = target_gate  # Set new target gate for inference

    #------------------------------------------------------------------------------------
    target_gate = np.array(target_gate)

    super_op_target_gate = (spre(Qobj(target_gate)) * spost(Qobj(target_gate))).data.toarray()
    U_target_dagger = np.array(super_op_target_gate.conjugate().transpose())
    #------------------------------------------------------------------------------------

    # print(f'U_target:\n{inference_env_config["U_target"]}\n\n')
    # inference_env_config['threshold_based_training'] = False
    inference_env_config['switch_every_episode'] = False
    inference_env_config['verbose'] = False
    inference_env = NoisySingleQubitEnv(inference_env_config)

    transition = inference_env.transition_history

    num_episodes = 0
    episode_reward = 0.0
    print("*************************************************************************************************")
    # print("Inferencing on a different gate is starting ....")
    print(f'Inference Gate: {inferencing_gate}\n')

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
        # print(f'Observation:\n{obs}\n')
        self_U = np.array(inference_env.get_self_U())
        fidelity = float(np.abs(np.trace(U_target_dagger @ self_U))) / (self_U.shape[0])
        calculated_target[fidelity] = self_U
        # print(f'Episode: {num_episodes}\nSelf.U:\n{final_gate_kron}')
        # Is the episode `done`? -> Reset.
        if done:
            # print(f"Episode done: Total reward = {episode_reward}")
            obs, info = inference_env.reset()
            num_episodes += 1
            episode_reward = 0.0

    inference_env_fidelity = [row[0] for row in inference_env.transition_history]
    max_fidel_idx = inference_env_fidelity.index(max(inference_env_fidelity))
    inference_self_U = [row[3] for row in inference_env.transition_history]
    best_inference_self_U = np.array(inference_self_U[max_fidel_idx])
    # best_target_dagger = best_inference_self_U.conjugate().transpose()
    # best_fidelity = float(np.abs(np.trace(U_target_dagger @ best_inference_self_U))) / (best_inference_self_U.shape[0])
    best_fidelity = float(np.abs(np.trace(U_target_dagger @ super_op_target_gate))) / (super_op_target_gate.shape[0])

    #
    # print(f'1)Super_op_target_gate:\n{super_op_target_gate}\n\n')
    # print(f'2)Final Self.U:\n{best_inference_self_U}\n\n')
    # print(f'Idx_gate: \n{inference_self_U[max_fidel_idx]}\n')
    # print(f'3)Best Fidelity: {best_fidelity}\n')
    # print(f'4)Fidelity (Best according to idx): {inference_env_fidelity[max_fidel_idx]}\n')
    # unitary_check1 = target_gate * (target_gate.conjugate().transpose())
    # unitary_check2 = (target_gate.conjugate().transpose()) * target_gate
    # print(f'\nIs the target unitary1:\n {unitary_check1}\n')
    # print(f'\nIs the target unitary2:\n {unitary_check2}\n')
    # print(f'Idx: {max_fidel_idx}')
    transition2 = inference_env.transition_history

    return inference_env, alg, calculated_target, target_gate


if __name__ == "__main__":
    # base_path = "/Users/vishchaudhary/rl-repo/results/2024-12-30_11-53-34/model_checkpoints"
    # base_path = "/Users/vishchaudhary/rl-repo/results/2024-12-30_14-56-24/model_checkpoints"
    # base_path = "/Users/vishchaudhary/rl-repo/results/2024-12-31_23-06-40/model_checkpoints"
    # base_path = "/Users/vishchaudhary/rl-repo/results/2025-01-09_00-21-26/model_checkpoints"
    base_path = "/Users/vishchaudhary/rl-repo/results/2024-12-29_16-35-02/model_checkpoints"

    # original_training_date = "2024-12-31_23-06-40"
    n_episodes_for_inferencing = 1

    save = True
    plot = True
    plot_target_change = False

    inferencing_gate = [gates.I(), gates.X(), gates.Z(), gates.H(), gates.RandomSU2(), gates.S(), gates.X_pi_4(),
                        gates.Y()]
    ##Error gates: S,X_pi_4,
    # inferencing_gate = [gates.I(), gates.X(), gates.Y()]
    continue_training(inference_gate=inferencing_gate,
                      n_episodes_for_inferencing=n_episodes_for_inferencing, save=save, plot=plot,
                      plot_target_change=plot_target_change)

    # target_name = gates.X()
    # for i in range(4):
    #
    #     print(f'Gate {i}')
    #     target = np.array(target_name.get_matrix())
    #     print(f'U_Target:\n{target}\n')
    #     unitary_check1 = np.dot(target,(target.conjugate().transpose()))
    #     unitary_check2 = (target.conjugate().transpose()) @ target
    #     print(f'U*U_dagger:\n{unitary_check1}\n')
    #     print(f'U_dagger*U:\n{unitary_check2}\n')

#Total List: I,X,Y,Z,H,S,X_pi_4,RandomSU2

#Is Unitary: I,X,Y,Z,H,RandomSU2
#Is not Unitary: S, X_pi_4
