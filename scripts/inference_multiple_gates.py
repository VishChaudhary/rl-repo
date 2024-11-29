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

def non_kron(K):
    m = int(np.sqrt(K.shape[0]))
    n = int(np.sqrt(K.shape[1]))

    # Recover a from the top-left block of K
    top_left_block = K[:m, :n]
    a_recovered = top_left_block / top_left_block[0, 0]

    return a_recovered

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

def createTextfile(save_filepath, inference_gate, final_gate, final_gate_nKron):
    # Example variables (replace these with actual values)
    self_U_target_nonKron = np.array([[0, 1], [1, 0]])
    target_gate = inference_gate.get_matrix()
    super_op_target_gate = np.kron(target_gate.conj(), target_gate)
    # File name based on xGate
    file_name = save_filepath + f"{inference_gate}_self_U.txt"

    # Write to file
    with open(file_name, "w") as f:
        f.write("self.U_kron:\n")
        f.write(f"{final_gate}\n\n")
        f.write("self.U_target_kron:\n")
        f.write(f"{super_op_target_gate}\n\n")
        f.write("self.U_nonKron:\n")
        f.write(f"{final_gate_nKron}\n\n")
        f.write("self.U_target_nonKron:\n")
        f.write(f"{target_gate}\n\n")

    # print(f"File '{file_name}' created successfully!")

def run(train_gate, inference_gate, n_training_iterations=1, n_episodes_for_inferencing = 1, save=True, plot=True, noise_file=noise_file):

    ray.init(num_cpus=2,   # change to your available number of CPUs
        num_gpus=0,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False)
    # env_config = GateSynthEnvRLlibHaarNoisy.get_default_env_config()
    # env = GateSynthEnvRLlibHaarNoisy(env_config)
    register_env("my_env", env_creator)

    # ---------------------> Configure algorithm and Environment <-------------------------
    # Initialize default configuration
    env_config = GateSynthEnvRLlibHaarNoisy.get_default_env_config()


    print(env_config)

    save_filepath = "/Users/vishchaudhary/rl-repo/results/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")

    # env_config["U_target"] = Gate.H
    # target_gate = gates.RandomSU2()
    env_config["U_target"] = train_gate.get_matrix()
    training_plot_filename = f'training_{train_gate}.png'


    # ---------------------> Get quantum noise data <-------------------------
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)

    env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(),
                                           np.reciprocal(t2_list).tolist()]  # using real T1 data
    env_config["detuning_list"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(), sigmaz()]
    # env_config["observation_space_size"] = 2*16 + 1 + 2 + 1 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate + 1 for detuning
    # env_config["observation_space_size"] = 36  # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate + 1 for detuning
    env_config["verbose"] = True

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    alg_config.environment("my_env", env_config=env_config)
    # alg_config.environment(GateSynthEnvRLlibHaarNoisy, env_config=GateSynthEnvRLlibHaarNoisy.get_default_env_config())
    alg_config.callbacks(GateSynthesisCallbacks)
    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.train_batch_size = GateSynthEnvRLlibHaarNoisy.get_default_env_config()["steps_per_Haar"]

    ### working 1-3 sets
    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"

    alg_config.num_steps_sampled_before_learning_starts = 1000

    # ---------------------> Original Parameters <-------------------------
    # alg_config.actor_lr = 4e-5
    # alg_config.critic_lr = 5e-4
    # alg_config.actor_hiddens = [30, 30, 30]

    # # ---------------------> Tuned Parameters <-------------------------
    # alg_config.actor_lr = 3.9e-5
    # alg_config.critic_lr = 7e-4
    # alg_config.actor_hiddens = [300] * 10
    # alg_config.critic_hiddens = [300] * 100

    # ---------------------> Tuned Parameters <-------------------------
    # alg_config.actor_lr = 2.69e-5
    # alg_config.critic_lr = 6.5e-4
    # alg_config.actor_hiddens = [300] * 10
    # alg_config.critic_hiddens = [50] * 100

    # # ---------------------> Sanya's Parameters <-------------------------
    alg_config.actor_lr = 2.65e-05
    alg_config.critic_lr = 9.35e-05
    alg_config.actor_hiddens = [300, 300, 300]
    alg_config.critic_hiddens = [50,50,50]


    alg_config.exploration_config["scale_timesteps"] = 10000
    print(f'\nalg_config:\n{alg_config}\n')
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
        inference_env, alg, final_gate_kron = do_inferencing(env, train_alg, inferencing_gate, n_episodes_for_inferencing)
        final_gate_nKron = non_kron(final_gate_kron)
        # ---------------------> Save/Plot Inference Results <-------------------------
        if save and plot is True:
            sr = SaveResults(inference_env, alg, save_path=save_filepath,
                             target_gate_string=f"Noisy_Train-{str(train_gate)}, Inference-{str(inferencing_gate)}")
            save_dir = sr.save_results()
            plot_data(save_dir, plot_filename=inferencing_plot_filename,
                      episode_length=alg._episode_history[0].episode_length,
                      figure_title=f"[NOISY] Inferencing on {str(inferencing_gate)} (Previously Trained on {str(train_gate)})")
            createTextfile(save_filepath, inferencing_gate, final_gate_kron, final_gate_nKron)
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

    # Initialize a new environment for inference using this configuration
    inference_env_config = env.return_env_config()
    inference_env_config["U_target"] = inferencing_gate.get_matrix()  # Set new target gate for inference
    print(f'U_target:\n{inference_env_config["U_target"]}\n\n')
    # inference_env_config["observation_space_size"] = 36  # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate + 1 for detuning
    inference_env = GateSynthEnvRLlibHaarNoisy(inference_env_config)

    num_episodes = 0
    episode_reward = 0.0
    print("Inferencing on a different gate is starting ....")
    print("*************************************************************************************************")
    # print("---------------------------------------------Initial Self.U---------------------------------------------")
    # print(inference_env.get_self_U())
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
        print(f'Episode: {num_episodes}\nSelf.U:\n{inference_env.get_self_U()}')
        final_gate_kron = np.array(inference_env.get_self_U())
        # Is the episode `done`? -> Reset.
        if done:
            print(f"Episode done: Total reward = {episode_reward}")
            obs, info = inference_env.reset()
            num_episodes += 1
            episode_reward = 0.0
    # print("---------------------------------------------Final Self.U---------------------------------------------")
    # print(inference_env.get_self_U())
    # print("------------------------------------------Inferencing Gate:-------------------------------------------")
    # print(inferencing_gate.get_matrix())
    # inference_gate_superop = np.kron(inferencing_gate.get_matrix().conj(), inferencing_gate.get_matrix())
    # print(f'Equivalent Super Operator Inference Gate:\n{inference_gate_superop}')
    # print("*************************************************************************************************")
    # print(type(final_gate_kron))
    return inference_env, alg, final_gate_kron


if __name__ == "__main__":
    n_training_iterations = 30
    n_episodes_for_inferencing = 10
    #Change the compute_fidelity func in the class
    #run only 1 inference episode and after it is done print the U_target for the inference and the self.U for the inference
    #you need to make
    save = True
    plot = True
    # a = gates.X_pi_4().get_matrix()
 #    K = [[ 9.99667544e-01+1.03397577e-25j,  6.22738935e-05+5.71575704e-05j, 6.22738935e-05-5.71575704e-05j,  7.14927843e-09+2.06795153e-25j],
 # [-1.10120249e-05+8.37937054e-05j,  7.65995347e-01-6.41984786e-01j, 4.10592396e-09+5.85101678e-09j,  1.10189550e-05-8.38211465e-05j],
 # [-1.10120249e-05-8.37937054e-05j,  4.10592396e-09-5.85101678e-09j, 7.65995347e-01+6.41984786e-01j,  1.10189550e-05+8.38211465e-05j],
 # [ 3.32456472e-04+1.00321841e-25j, -6.22738935e-05-5.71575704e-05j, -6.22738935e-05+5.71575704e-05j,  9.99999993e-01-1.00288802e-25j]]
 #    K = np.array(K)
 #    # K = np.kron(a.conj(),a)
 #
    # Infer dimensions of a from K
    # m = int(np.sqrt(K.shape[0]))
    # n = int(np.sqrt(K.shape[1]))
    #
    # # Recover a from the top-left block of K
    # top_left_block = K[:m, :n]
    # a_recovered = top_left_block / top_left_block[0, 0]
    # print(f'X_pi_4:\n{gates.X_pi_4().get_matrix()}\n')
    # # print(f'a original:\n{a}\n')
    # print(f'a_recovered:\n{a_recovered}\n')

    train_gate = gates.RandomSU2()
    inferencing_gate = [gates.X_pi_4()]
    # inferencing_gate = [gates.X_pi_4(), gates.X(), gates.Y(), gates.Z(), gates.H(), gates.S(), gates.RandomSU2(), gates.I()]
    run(train_gate, inferencing_gate, n_training_iterations, n_episodes_for_inferencing, save, plot, noise_file)

