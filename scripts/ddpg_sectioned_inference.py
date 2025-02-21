import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.DDPG_noisy_single_qubit_env import DDPG_NoisySingleQubitEnv
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
from relaqs.api.utils import *

noise_file = "april/ibmq_belem_month_is_4.json"

def training(training_gates, n_training_episodes):

    ray.init(num_cpus=14,  # change to your available number of CPUs
             num_gpus=20,
             include_dashboard=False,
             ignore_reinit_error=True,
             log_to_driver=False)

    env_config = DDPG_NoisySingleQubitEnv.get_default_env_config()
    # ---------------------> Get quantum noise data <-------------------------
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)

    # ---------------------> Configure Environment <-------------------------
    env_config['num_Haar_basis'] = 1
    env_config['steps_per_Haar'] = 2

    env_config["U_target"] = training_gates[0].get_matrix()
    env_config["training"] = False
    env_config["retraining"] = True
    env_config["retraining_gates"] = training_gates
    env_config["verbose"] = False

    env_config["relaxation_rates_list"] = [t1_list, t2_list]  # using real T1 data
    env_config["detuning_list"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(), sigmaz()]

    # ---------------------> Configure algorithm<-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    alg_config.environment(DDPG_NoisySingleQubitEnv, env_config=env_config)
    alg_config.callbacks(GateSynthesisCallbacks)
    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.train_batch_size = 256

    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"

    # # ---------------------> Tuned Parameters <-------------------------

    alg_config.actor_hiddens = [1024, 512, 256]
    alg_config.actor_lr = 5e-5
    alg_config.critic_lr = 1e-3
    alg_config.critic_hiddens = [1024, 512, 300]

    alg_config.exploration_config["random_timesteps"] = 3055.8304716435505
    alg_config.exploration_config["ou_base_scale"] = 0.33536897625927453
    alg_config.exploration_config["ou_theta"] = 0.31360827370009975
    alg_config.exploration_config["ou_sigma"] = 0.26940347674578985
    alg_config.exploration_config["initial_scale"] = 1.1
    alg_config.exploration_config["scale_timesteps"] = n_training_episodes*1000
    # alg_config.exploration_config["scale_timesteps"] = 75000
    # alg_config.replay_buffer_config["capacity"] = 200000
    alg_config.target_network_update_freq = 2  # Slows down updates slightly for stability
    # alg_config.tau = 0.0025  # Makes each target update more meaningful
    alg_config.tau = 0.0005
    alg_config.twin_q = True

    alg = alg_config.build()

    n_training_episodes *= env_config['num_Haar_basis'] * env_config['steps_per_Haar']

    update_every_percent = 2
    results = []
    update_interval = max(1, int(n_training_episodes * (update_every_percent / 100)))

    training_start_time = get_time()

    for i in range(n_training_episodes):
        results.append(alg.train())
        # Print update every x%
        if (i + 1) % int(update_interval) == 0 or (i + 1) == n_training_episodes:
            percent_complete = (i + 1) / n_training_episodes * 100
            print(f"Training Progress: {percent_complete:.0f}% complete")

    training_end_time = get_time()
    training_elapsed_time = training_end_time - training_start_time

    return alg, training_elapsed_time


def save_training_results(training_alg, filepath, filename, title_figure):
    train_env = training_alg.workers.local_worker().env
    env_config = train_env.return_env_config()
    alg_config = training_alg.workers.local_worker().config

    # ---------------------> Save/Plot Training Results <-------------------------
    sr = SaveResults(train_env, training_alg, save_path=filepath)
    save_dir = sr.save_results()
    plot_data(save_dir, plot_filename=filename,
              figure_title=f"[NOISY] Training on {title_figure}", gate_switch_array=None)
    print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    config_table(env_config=env_config, alg_config=alg_config, filepath=filepath)


def inference_and_save(inference_list, save_dir, train_alg, n_episodes_for_inferencing):
    columns = ['Fidelity', 'Rewards', 'Actions', 'Self.U_Operator', 'U_target', 'Episode Id']
    inf_count = 0

    for curr_gate in inference_list:
        # train_alg = copy.deepcopy(alg)

        gate_save_dir = save_dir + f'/{curr_gate}/'
        plot_filename = f'inference_{curr_gate}.png'
        os.makedirs(gate_save_dir)

        if inf_count == 0:
            figure_title = f"[NOISY] Inferencing on Multiple Different {str(curr_gate)}."
        else:
            figure_title = f"[NOISY] Multiple Instances of Inferencing on {str(curr_gate)}"

        env_data_title = f"{curr_gate}_"
        transition_history = []

        for inference_iteration in range(n_episodes_for_inferencing):
            # -----------------------> Inferencing <---------------------------
            env = train_alg.workers.local_worker().env
            inference_env, target_gate, history = do_inferencing(env, train_alg, curr_gate)
            transition_history.append(history)

        df = pd.DataFrame(transition_history, columns=columns)
        # df.to_pickle(env_data_title + "env_data.pkl")  # easier to load than csv
        df.to_csv(gate_save_dir + env_data_title + "env_data.csv", index=False)  # backup in case pickle doesn't work
        plot_multiple_visuals(df, figure_title=figure_title, save_dir=gate_save_dir, plot_filename=plot_filename,
                              inf_count=inf_count, gate=curr_gate)

        inf_count += 1


def do_inferencing(env, train_alg, curr_gate):
    """
    alg: The trained model
    n_episodes_for_inferencing: Number of episodes to do during the training
    """

    # Initialize a new environment for inference using this configuration
    inference_env_config = env.return_env_config()
    target_gate = curr_gate.get_matrix()  # Set new target gate for inference

    inference_env_config["U_target"] = target_gate
    inference_env_config['training'] = False
    inference_env_config['verbose'] = False
    inference_env_config['retraining'] = False
    env_class = type(env)
    inference_env = env_class(inference_env_config)

    # ------------------------------------------------------------------------------------
    target_gate = np.array(target_gate)

    episode_reward = 0.0
    done = False

    ##Check this out. When we reset it computes the fidelity for the observation between the U_inital (I gate by default) and our new target for the first haar num and step per haar
    obs, info = inference_env.reset()  # Start with the inference environment
    while not done:

        # Compute an action (`a`).
        action = train_alg.compute_single_action(
            observation=obs,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `action` to the env.
        obs, reward, done, truncated, _ = inference_env.step(action)
        episode_reward += reward

        if done:
            return inference_env, target_gate, inference_env.transition_history[-1]


def main():
    save_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
    save_filepath = "/Users/vishchaudhary/rl-repo/results/" + save_date

    # training_gates = [gates.RandomSU2(theta_range=(0,0.58))]
    a = gates.Ry(lower_bound=1.20, upper_bound=1.8)
    b = gates.Ry(lower_bound=0.20, upper_bound=0.8)
    c = gates.Rx(lower_bound=0.2, upper_bound=0.8)
    d = gates.Rx(lower_bound=1.2, upper_bound=1.8)
    e = gates.Rz(lower_bound=0.2, upper_bound=0.8)
    f = gates.Rz(lower_bound=1.2, upper_bound=1.8)
    training_gates = [a,b,c,d,e,f]
    training_name = ""

    for gate in training_gates:
        training_name += f"{gate}_"

    training_plot_filename = f'{training_name}_training.png'

    # Modified to be number of episodes for training (in thousands) for EACH retrain gate
    n_training_iterations = 75
    n_episodes_for_inferencing = 1000

    # RandomGate, Rx, and Ry must be kept as the first 3 gates (order is irrelevant) in order to plot their bloch sphere
    inferencing_gate = [gates.RandomSU2(), gates.Rx(), gates.Ry(), gates.Rz(),
                        gates.X(), gates.Y(), gates.Z(), gates.H(), gates.S(),gates.HS(),gates.XY_combination(),gates.ZX_combination()]


    alg, training_time = training(training_gates, n_training_iterations)

    save_training_results(training_alg=alg, filepath=save_filepath,
                          filename=training_plot_filename,
                          title_figure=training_name)

    inference_and_save(inference_list=inferencing_gate, save_dir=save_filepath, train_alg=alg,
                       n_episodes_for_inferencing = n_episodes_for_inferencing)

    print(f"Training Elapsed time: \n{training_time}\n")
    print(f'Results saved to: {save_date}')

# Uses the idea of retraining on just the poorly performing Pauli Gates to see if that will boost overall performance
# without hurting the inferencing of the other gates

if __name__ == "__main__":
    main()






