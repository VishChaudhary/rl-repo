""" Learning new single qubit gates, rather than the default X gate. """

import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
import relaqs.api.gates as gates
import numpy as np
from relaqs.api.callbacks import GateSynthesisCallbacks
import os
import datetime
from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj, tensor
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

def createTextfile(save_filepath, inference_gate, final_gate):
    # Example variables (replace these with actual values)
    target_gate = inference_gate.get_matrix()
    super_op_target_gate = (spre(Qobj(target_gate)) * spost(Qobj(target_gate))).data.toarray()
    file_name = save_filepath + f"{inference_gate}_self_U.txt"
    U_target_dagger = super_op_target_gate.conjugate().transpose()

    fidelity = float(np.abs(np.trace(U_target_dagger @ final_gate))) / (final_gate.shape[0])

    # Write to file
    with open(file_name, "w") as f:
        f.write("self.U:\n")
        f.write(f"{final_gate}\n\n")
        f.write("self.U_target:\n")
        f.write(f"{super_op_target_gate}\n\n")
        f.write(f"Fidelity:\n")
        f.write(f"{fidelity}\n")


def run(train_gate, inference_gate, n_training_iterations=1, n_episodes_for_inferencing = 1, save=True, plot=True, noise_file=noise_file):

    ray.init(num_cpus=2,   # change to your available number of CPUs
        num_gpus=0,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False)

    # ---------------------> Configure algorithm and Environment <-------------------------
    # Initialize default configuration
    env_config = GateSynthEnvRLlibHaarNoisy.get_default_env_config()

    save_filepath = "/Users/vishchaudhary/rl-repo/results/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
    training_plot_filename = f'training_{train_gate}.png'


    # ---------------------> Get quantum noise data <-------------------------
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)

    # ---------------------> Configure Environment <-------------------------
    env_config["U_target"] = train_gate.get_matrix()
    env_config['num_Haar_basis'] = 1
    env_config['steps_per_Haar'] = 2
    env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(),
                                           np.reciprocal(t2_list).tolist()]  # using real T1 data
    env_config["detuning_list"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(), sigmaz()]
    env_config["fidelity_threshold"] = 0.4
    env_config["fidelity_target_switch_case"] = 1
    env_config["base_target_switch_case"] = 7000
    env_config["verbose"] = True

    # ---------------------> Configure algorithm<-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    alg_config.environment(GateSynthEnvRLlibHaarNoisy, env_config=env_config)
    alg_config.callbacks(GateSynthesisCallbacks)
    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.train_batch_size = env_config["steps_per_Haar"]

    ### working 1-3 sets
    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    alg_config.num_steps_sampled_before_learning_starts = 1000

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
    # alg_config.exploration_config["scale_timesteps"] = 19885.54898737561
    alg_config.exploration_config["scale_timesteps"] = 10123.97829415627

    # ---------------------> Close  Second Fidelity/Reward Parameters <-------------------------
    # alg_config.exploration_config["random_timesteps"] = 4673.765975569726
    # alg_config.exploration_config["ou_base_scale"] = 0.1367000631272562
    # alg_config.exploration_config["ou_theta"] = 0.13792729799506298
    # alg_config.exploration_config["ou_sigma"] = 0.11462192187335964
    # alg_config.exploration_config["initial_scale"] = 0.5497884198832642
    # alg_config.exploration_config["scale_timesteps"] = 10123.97829415627

    print(f'\nalg_config:\n{alg_config}\n')
    alg = alg_config.build()
    # ---------------------------------------------------------------------


    # ---------------------> Train Agent <-------------------------
    results = [alg.train() for _ in range(n_training_iterations)]
    # -------------------------------------------------------------

    train_alg = alg
    train_env = train_alg.workers.local_worker().env
    num_steps_done = train_env.get_self_episode_num()

    # ---------------------> Save/Plot Training Results <-------------------------
    if save and plot is True:
        # train_env = train_alg.workers.local_worker().env
        sr = SaveResults(train_env, train_alg, save_path=save_filepath,
                         target_gate_string=f"Noisy_Train-{str(train_gate)}, Inference-{str(inference_gate)}")
        save_dir = sr.save_results()
        plot_data(save_dir, plot_filename=training_plot_filename,
                  episode_length=train_alg._episode_history[0].episode_length,
                  figure_title=f"[NOISY] Training on {str(train_gate)}")
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    for inferencing_gate in inference_gate:
        inferencing_plot_filename = f'inferencing_{inferencing_gate}.png'
        # -----------------------> Inferencing <---------------------------
        env = train_alg.workers.local_worker().env
        inference_env, alg, final_gate_kron = do_inferencing(env, train_alg, inferencing_gate, n_episodes_for_inferencing)

        # ---------------------> Save/Plot Inference Results <-------------------------
        if save and plot is True:
            sr = SaveResults(inference_env, alg, save_path=save_filepath,
                             target_gate_string=f"Noisy_Train-{str(train_gate)}, Inference-{str(inferencing_gate)}")
            save_dir = sr.save_results()
            plot_data(save_dir, plot_filename=inferencing_plot_filename,
                      episode_length=alg._episode_history[0].episode_length,
                      figure_title=f"[NOISY] Inferencing on {str(inferencing_gate)} (Previously Trained on {str(train_gate)})")
            createTextfile(save_filepath, inferencing_gate, final_gate_kron)
            print("Results saved to:", save_dir)
        # --------------------------------------------------------------
    print(f'Num times steps called: {num_steps_done}\nWhile training for training iterations num: {n_training_iterations}\n')
    num_haar_basis = env_config['num_Haar_basis']
    num_steps_haar = env_config['steps_per_Haar']
    print(f'Num Haar basis: {num_haar_basis}\nNum steps per haar: {num_steps_haar}')




def do_inferencing(env, alg, inferencing_gate, n_episodes_for_inferencing):
    """
    alg: The trained model
    n_episodes_for_inferencing: Number of episodes to do during the training
    """

    assert n_episodes_for_inferencing > 0

    # Initialize a new environment for inference using this configuration
    inference_env_config = env.return_env_config()
    inference_env_config["U_target"] = inferencing_gate.get_matrix()  # Set new target gate for inference
    print(f'U_target:\n{inference_env_config["U_target"]}\n\n')
    inference_env = GateSynthEnvRLlibHaarNoisy(inference_env_config)

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
        obs, reward, done, truncated, _ = inference_env.step(action, False)
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
    n_training_iterations = 75
    n_episodes_for_inferencing = 100

    save = True
    plot = True

    train_gate = gates.RandomSU2()

    inferencing_gate = [gates.X_pi_4(), gates.X(), gates.Y(), gates.Z(), gates.H(), gates.S(), gates.RandomSU2(), gates.I()]
    run(train_gate, inferencing_gate, n_training_iterations, n_episodes_for_inferencing, save, plot, noise_file)

