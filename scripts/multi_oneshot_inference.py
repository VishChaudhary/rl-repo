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


def oneshot_inference(alg = None, inference_gate = None, original_training_date = None):
    # save_filepath = "/Users/vishchaudhary/rl-repo/results/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
    # training_plot_filename = f'training_{train_gate}.png'


    env = alg.workers.local_worker().env


    # Initialize a new environment for inference using this configuration
    inference_env_config = env.return_env_config()
    target_gate = inference_gate.get_matrix()
    inference_env_config["U_target"] = target_gate  # Set new target gate for inference
    inference_env_config['switch_every_episode'] = False
    inference_env_config['verbose'] = False
    inference_env = NoisySingleQubitEnv(inference_env_config)

    # ------------------------------------------------------------------------------------
    target_gate = np.array(target_gate)

    super_op_target_gate = (spre(Qobj(target_gate)) * spost(Qobj(target_gate))).data.toarray()
    U_target_dagger = np.array(super_op_target_gate.conjugate().transpose())
    # ------------------------------------------------------------------------------------

    num_episodes = 0
    episode_reward = 0.0
    print("*************************************************************************************************")
    print(f'Inference Gate: {inference_gate}\n')

    obs, info = inference_env.reset()  # Start with the inference environment
    calculated_target = {}
    while num_episodes < 1:
        # print("Episode : ", num_episodes)
        # Compute an action (`a`).
        action = alg.compute_single_action(
            observation=obs,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `action` to the env.
        obs, reward, done, truncated, _ = inference_env.step(action)
        episode_reward += reward
        self_U = np.array(inference_env.get_self_U())
        fidelity = float(np.abs(np.trace(U_target_dagger @ self_U))) / (self_U.shape[0])
        calculated_target[fidelity] = self_U
        # Is the episode `done`? -> Reset.
        if done:
            # print(f"Episode done: Total reward = {episode_reward}")
            obs, info = inference_env.reset()
            num_episodes += 1
            episode_reward = 0.0


    print(calculated_target)
    ray.shutdown()

    # return inference_env, alg, calculated_target, target_gate






if __name__ == "__main__":
    base_path = "/Users/vishchaudhary/rl-repo/results/2024-12-30_11-53-34/model_checkpoints"
    # base_path = "/Users/vishchaudhary/rl-repo/results/2024-12-30_14-56-24/model_checkpoints"
    # base_path = "/Users/vishchaudhary/rl-repo/results/2024-12-31_23-06-40/model_checkpoints"
    # base_path = "/Users/vishchaudhary/rl-repo/results/2025-01-09_00-21-26/model_checkpoints"
    # base_path = "/Users/vishchaudhary/rl-repo/results/2024-12-29_16-35-02/model_checkpoints"
    alg = load_model(base_path)

    # original_training_date = "2024-12-31_23-06-40"
    n_iterations_of_inferencing = 10

    save = True
    plot = True
    plot_target_change = False

    inferencing_gate = [gates.I(), gates.X(), gates.Z(), gates.H(), gates.RandomSU2(), gates.S(), gates.X_pi_4(),
                        gates.Y()]
    inference_gate = gates.Z()
    oneshot_inference(alg=alg, inference_gate=inference_gate)

