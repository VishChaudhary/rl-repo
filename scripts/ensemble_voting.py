from collections import defaultdict
from relaqs.environments.noisy_single_qubit_env import NoisySingleQubitEnv
import ray
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
import relaqs.api.gates as gates
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from datetime import datetime
from qutip.operators import *
import qutip
from relaqs.api.utils import *


# def calculate_fidelity(target_gate, generated_unitary):
#     # print(f'Calculating fidelity between {target_gate}\n and \n{generated_unitary}')
#     moded_target = (spre(Qobj(target_gate)) * spost(Qobj(target_gate))).data.toarray()
#     U_target_dagger = moded_target.conjugate().transpose()
#     fidelity = float(np.abs(np.trace(U_target_dagger @ generated_unitary))) / (generated_unitary.shape[0])
#     return fidelity

def ensemble_voting(alg_list, inference_list,  n_episodes_for_inferencing):
    #
    each_alg_stats = defaultdict(lambda: defaultdict(int))
    overall_alg_stats = defaultdict(int)

    save_dir = "/Users/vishchaudhary/rl-repo/results/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    columns = ['Fidelity', 'Rewards', 'Actions', 'Self.U_Operator', 'U_target', 'Episode Id']
    inf_count = 0

    for gate in inference_list:

        gate_save_dir = save_dir + f'/{gate}/'
        plot_filename = f'inference_{gate}.png'
        os.makedirs(gate_save_dir)

        if inf_count < 2:
            figure_title = f"[NOISY] Inferencing on Multiple Different {str(gate)}"
        else:
            figure_title = f"[NOISY] Inferencing on Multiple Instances of {str(gate)}"

        env_data_title = f"{gate}_"
        transition_history = []

        for inference_iteration in range(n_episodes_for_inferencing):
            highest_fidelity = float("-inf")
            best_alg_index = None
            best_history = None

            target_gate = gate.get_matrix()


            for alg_id, alg in enumerate(alg_list):
                # -----------------------> Inferencing <---------------------------
                generated_unitary, history = inference(alg, target_gate)
                fidelity = history[0]
                # print(f'History: {history}\n')
                # print(f'Fidelity: {history[0]}\n')
                # fidelity = calculate_fidelity(target_gate, generated_unitary)
                if fidelity > highest_fidelity:
                    highest_fidelity = fidelity
                    best_alg_index = alg_id
                    best_history = history

            overall_alg_stats[best_alg_index] += 1
            each_alg_stats[best_alg_index][gate] += 1
            transition_history.append(best_history)

        # print(f'Transition History: \n{transition_history}\n')
        df = pd.DataFrame(transition_history, columns=columns)
        # df.to_pickle(env_data_title + "env_data.pkl")  # easier to load than csv
        df.to_csv(gate_save_dir + env_data_title + "env_data.csv", index=False)  # backup in case pickle doesn't work
        plot_multiple_visuals(df, figure_title=figure_title, save_dir=gate_save_dir, plot_filename=plot_filename,
                              inf_count=inf_count, gate=gate)

        inf_count += 1

    #Create overall stats plot for all the algs
    plot_overall_alg_votes(overall_alg_stats, save_dir=save_dir)
    #Create a subplot for each alg showing how many times it was called
    plot_alg_gate_counts(each_alg_stats, save_dir)
    #Write to a text file exactly what algs were used (the dates of the files)




    print(f'File saved to: {save_dir}')
    return save_dir

    ray.shutdown()


def inference(alg, target_gate):
    """
    alg: The trained model
    n_episodes_for_inferencing: Number of episodes to do during the training
    """

    # Initialize a new environment for inference using this configuration
    env = alg.workers.local_worker().env
    inference_env_config = env.return_env_config()

    inference_env_config["U_target"] = target_gate
    inference_env_config['training'] = False
    inference_env_config['verbose'] = False
    inference_env_config['retraining'] = False
    inference_env = NoisySingleQubitEnv(inference_env_config)

    # ------------------------------------------------------------------------------------
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
            generated_unitary = inference_env.get_self_U()
            return generated_unitary, inference_env.transition_history[-1]


def main() :
    checkpoint_dates = ["2025-02-06_16-49-10", "2025-02-06_14-27-50", "2025-02-06_21-09-57", "2025-02-06_19-51-54","2025-02-07_13-24-27"]
    alg_list = []

    n_episodes_for_inferencing = 1000

    ##RandomGate must be kept as first in the array and XY_combination MUST be kept as second in the array
    inference_list = [gates.RandomSU2(), gates.XY_combination(), gates.X(), gates.Y(), gates.Z(),
                        gates.H(), gates.S(),
                        gates.X_pi_4()]

    for date in checkpoint_dates:
        model_checkpoint = "/Users/vishchaudhary/rl-repo/results/" + date + "/model_checkpoints"
        alg_list.append(load_model(model_checkpoint))

    save_dir = ensemble_voting(alg_list, inference_list, n_episodes_for_inferencing)
    save_alg_names_to_file(checkpoint_dates, save_dir)


if __name__ == "__main__":
    main()

