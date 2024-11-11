""" Learning new single qubit gates, rather than the default X gate. """

import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
import relaqs.api.gates as gates
import numpy as np
import os
import datetime
from qutip.operators import *
from relaqs.quantum_noise_data.get_data import (get_month_of_all_qubit_data,
get_single_qubit_detuning)

# path_to_relaqs_root = os.path.dirname(os.path.realpath(__file__))
path_to_relaqs_root = '/Users/vishchaudhary/rl-repo/src/relaqs'
QUANTUM_NOISE_DATA_DIR = path_to_relaqs_root + "/quantum_noise_data/"
noise_file = "april/ibmq_belem_month_is_4.json"
# inferencing_noise_file = "april/ibmq_manila_month_is_4.json"

def env_creator(config):
    return GateSynthEnvRLlibHaarNoisy(config)

def sample_noise_parameters(t1_t2_noise_file, detuning_noise_file = None):
    # ---------------------> Get quantum noise data <-------------------------
    t1_list, t2_list = get_month_of_all_qubit_data(QUANTUM_NOISE_DATA_DIR + t1_t2_noise_file)        #in seconds

    if detuning_noise_file is None:
        mean = 0
        std = 0.03
        sample_size = 100
        samples = np.random.normal(mean, std, sample_size)
        detunings = samples.tolist()
    else:
        detunings = get_single_qubit_detuning(QUANTUM_NOISE_DATA_DIR + detuning_noise_file)

    return t1_list, t2_list, detunings


def run(train_gate, inference_gate, n_training_iterations=1, n_episodes_for_inferencing = 1, save=True, plot=True, noise_file=noise_file):

    ray.init()
    register_env("my_env", env_creator)

    # ---------------------> Configure algorithm and Environment <-------------------------


    env_config = GateSynthEnvRLlibHaarNoisy.get_default_env_config()

    save_filepath = "/Users/vishchaudhary/rl-repo/results/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")

    # env_config["U_target"] = Gate.H
    # target_gate = gates.RandomSU2()
    env_config["U_target"] = train_gate.get_matrix()
    training_plot_filename = f'training_{train_gate}.png'


    # ---------------------> Get quantum noise data <-------------------------
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)

    env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(),
                                           np.reciprocal(t2_list).tolist()]  # using real T1 data
    env_config["delta"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(), sigmaz()]
    # env_config["observation_space_size"] = 2*16 + 1 + 2 + 1 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate + 1 for detuning
    env_config["observation_space_size"] = 36  # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate + 1 for detuning
    env_config["verbose"] = True

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    alg_config.environment("my_env", env_config=env_config)
    # alg_config.environment(GateSynthEnvRLlibHaarNoisy, env_config=GateSynthEnvRLlibHaarNoisy.get_default_env_config())

    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.train_batch_size = GateSynthEnvRLlibHaarNoisy.get_default_env_config()["steps_per_Haar"]

    ### working 1-3 sets
    alg_config.actor_lr = 4e-5
    alg_config.critic_lr = 5e-4

    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    alg_config.num_steps_sampled_before_learning_starts = 1000
    alg_config.actor_hiddens = [30, 30, 30]
    alg_config.exploration_config["scale_timesteps"] = 10000

    alg = alg_config.build()
    # ---------------------------------------------------------------------


    # ---------------------> Train Agent <-------------------------
    for _ in range(n_training_iterations):
        result = alg.train()
    # -------------------------------------------------------------

    train_alg = alg

    for inferencing_gate in inference_gate:
        inferencing_plot_filename = f'inferencing_{inferencing_gate}.png'
        # -----------------------> Inferencing <---------------------------
        env = train_alg.workers.local_worker().env
        # print("Episode history length:", len(alg._episode_history))
        inference_env, alg = do_inferencing(env, train_alg, inferencing_gate, n_episodes_for_inferencing)
        # print("Episode history length:", len(alg._episode_history))
        # ---------------------> Save/Plot Inference Results <-------------------------
        if save and plot is True:
            sr = SaveResults(inference_env, alg, save_path=save_filepath,
                             target_gate_string=f"Noisy_Train-{str(train_gate)}, Inference-{str(inferencing_gate)}")
            save_dir = sr.save_results()
            plot_data(save_dir, plot_filename=inferencing_plot_filename,
                      episode_length=alg._episode_history[0].episode_length,
                      figure_title=f"[NOISY] Inferencing on {str(inferencing_gate)} (Previously Trained on {str(train_gate)})")
            print("Results saved to:", save_dir)
        # --------------------------------------------------------------


    # ---------------------> Save/Plot Training Results <-------------------------
    if save and plot is True:
        sr = SaveResults(env, train_alg, save_path = save_filepath, target_gate_string= f"Noisy_Train-{str(train_gate)}, Inference-{str(inference_gate)}")
        save_dir = sr.save_results()
        plot_data(save_dir, plot_filename=training_plot_filename,
                  episode_length=train_alg._episode_history[0].episode_length,
                  figure_title=f"[NOISY] Training on {str(train_gate)}")
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------


    # # ---------------------> Plot Training/Inferencing Data <-------------------------
    # if plot is True:
    #     assert save is True, "If plot=True, then save must also be set to True"
    #     plot_data(save_dir, plot_filename= training_plot_filename, episode_length=train_alg._episode_history[0].episode_length, figure_title=f"[NOISY] Training on {str(train_gate)}")
    #     plot_data(save_dir, plot_filename= inferencing_plot_filename, episode_length=alg._episode_history[0].episode_length, figure_title=f"[NOISY] Trained on {str(train_gate)}, Inference on {str(inference_gate)}")
    #     print("Plots Created")
    # # --------------------------------------------------------------


def do_inferencing(env, alg, inferencing_gate, n_episodes_for_inferencing):
    """
    alg: The trained model
    n_episodes_for_inferencing: Number of episodes to do during the training
    """

    assert n_episodes_for_inferencing > 0

    # Create a new inference environment configuration
    # inference_env_config = env.get_default_env_config()
    # inference_env_config = {
    #     "action_space_size": env.action_space.shape[0],
    #     "U_initial": env.U_initial,
    #     "U_target": inferencing_gate.get_matrix(),  # New target gate for inference
    #     "final_time": env.final_time,
    #     "num_Haar_basis": env.num_Haar_basis,
    #     "steps_per_Haar": env.steps_per_Haar,
    #     "delta": env.delta,
    #     "save_data_every_step": 1,
    #     "verbose": env.verbose,
    #     "relaxation_rates_list": env.relaxation_rates_list,
    #     "relaxation_ops": env.relaxation_ops,
    #     "observation_space_size": env.observation_space.shape[0],
    # }



    # Initialize a new environment for inference using this configuration
    inference_env_config = env.return_env_config()
    inference_env_config["U_target"] = inferencing_gate.get_matrix()  # Set new target gate for inference
    inference_env_config["observation_space_size"] = 36  # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate + 1 for detuning
    # inference_env_config["action_space_size"] = env.observation_space.shape[0]
    inference_env = GateSynthEnvRLlibHaarNoisy(inference_env_config)

    # Reinitialize the environment with the new target gate
    # inference_env = GateSynthEnvRLlibHaarNoisy(inference_env_config)

    num_episodes = 0
    episode_reward = 0.0
    print("Inferencing on a different gate is starting ....")

    obs, info = inference_env.reset()  # Start with the inference environment

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
        # Is the episode `done`? -> Reset.
        if done:
            print(f"Episode done: Total reward = {episode_reward}")
            obs, info = inference_env.reset()
            num_episodes += 1
            episode_reward = 0.0
    return inference_env, alg


if __name__ == "__main__":
    # n_training_iterations = 500
    n_training_iterations = 25
    n_episodes_for_inferencing = 1000
    save = True
    plot = True
    train_gate = gates.RandomSU2()
    inferencing_gate = [gates.X_pi_4(), gates.X(), gates.Y(), gates.Z(), gates.H(), gates.S(), gates.RandomSU2(), gates.I()]
    run(train_gate, inferencing_gate, n_training_iterations, n_episodes_for_inferencing, save, plot, noise_file)

