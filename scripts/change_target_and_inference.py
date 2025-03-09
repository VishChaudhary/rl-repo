import ray
import numpy as np
from ray.rllib.algorithms.ddpg import DDPGConfig
from relaqs.environments.changing_target_gate import ChangingTargetEnv, NoisyChangingTargetEnv
from relaqs.save_results import SaveResults
from relaqs.plot_data import *
from relaqs.api import gates
from relaqs.api.utils import *

# ibm_nairobi data chosen because sample_noise_parameter function defaults to ibm_nairobi machine + data for qubit label 1 unless otherwise specified
t1_t2_noise_file = "april/ibm_nairobi_month_is_4.json"

# detuning data
detuning_noise_file = "qubit_detuning_data.json"

def run(env=ChangingTargetEnv, n_training_episodes=1, u_target_list = [gates.RandomSU2()], save=True, plot=True):
    ray.init(num_cpus=14)

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    env_config = env.get_default_env_config()
    # env_config["target_generation_function"] = XY_combination
    env_config['num_Haar_basis'] = 1
    env_config['steps_per_Haar'] = 2
    env_config["U_target_list"] = u_target_list
    env_config["verbose"] = False

    if env == NoisyChangingTargetEnv:
        # ---------------------> Get quantum noise data <-------------------------
        print("Noisy Changing Target Noise Initialized")
        t1_list, t2_list, detuning_list = sample_noise_parameters(t1_t2_noise_file, detuning_noise_file)
        env_config["relaxation_rates_list"] = [t1_list, t2_list]  # using real T1 data
        env_config["detuning_list"] = detuning_list


    alg_config.environment(env, env_config=env_config)

    alg_config.rollouts(batch_mode="complete_episodes")

    #---------------------------------Configs---------------------------------
    ### working 1-3 sets
    # alg_config.actor_lr = 7e-5
    # alg_config.critic_lr = 2e-4
    #
    # alg_config.actor_hidden_activation = "relu"
    # alg_config.critic_hidden_activation = "relu"
    # alg_config.num_steps_sampled_before_learning_starts = 500
    # alg_config.actor_hiddens = [300, 300, 300, 300, 300]
    # alg_config.critic_hiddens = [400, 400, 300]
    # # alg_config.exploration_config["scale_timesteps"] = 1000
    # alg_config.train_batch_size = 256
    #
    # alg_config.tau = 0.0002
    # alg_config.target_network_update_freq = 4
    # alg_config.grad_clip = 10.0
    #
    # alg_config.exploration_config = {
    #     "type": "OrnsteinUhlenbeckNoise",
    #     "random_timesteps": 1000,
    #     "initial_scale": 1.0,  # Start with full noise
    #     "final_scale": 0.1,  # End with zero noise
    #     "scale_timesteps": 50000,  # Over 30,000 timesteps (adjust as needed)
    #     "ou_base_scale": 0.1,  # Base OU noise amplitude
    #     "ou_theta": 0.15,  # OU theta
    #     "ou_sigma": 0.2,  # OU sigma
    # }
    alg_config["actor_hiddens"] = [400, 400, 300, 300]
    alg_config["critic_hiddens"] = [400, 400, 300]
    alg_config["actor_hidden_activation"] = "relu"
    alg_config["critic_hidden_activation"] = "relu"

    # 2. Learning Rates
    alg_config["actor_lr"] = 8e-5
    alg_config["critic_lr"] = 3e-4

    # 3. Warm-Up and Batch Size
    alg_config["num_steps_sampled_before_learning_starts"] = 1000
    alg_config["train_batch_size"] = 256

    # 4. Target Network Updates & Stabilization
    alg_config["tau"] = 0.0002
    alg_config["target_network_update_freq"] = 4
    alg_config["grad_clip"] = 10.0

    # 5. Exploration (Ornstein-Uhlenbeck Noise)
    alg_config["exploration_config"] = {
        "type": "OrnsteinUhlenbeckNoise",
        "random_timesteps": 2000,
        "initial_scale": 1.0,
        "final_scale": 0.05,
        "scale_timesteps": 100000,
        "ou_base_scale": 0.1,
        "ou_theta": 0.15,
        "ou_sigma": 0.2,
    }
    # # 1) Larger Networks for More Expressive Power
    # alg_config["actor_hiddens"] = [600, 600, 400, 300]
    # alg_config["critic_hiddens"] = [600, 600, 400]
    # alg_config["actor_hidden_activation"] = "relu"
    # alg_config["critic_hidden_activation"] = "relu"
    #
    # # 2) Learning Rates
    # alg_config["actor_lr"] = 1e-4  # Slightly higher than 8e-5 for faster final adaptation
    # alg_config["critic_lr"] = 2e-4  # Lower than 3e-4 to reduce Q-value oscillations
    #
    # # 3) Training Setup
    # alg_config["num_steps_sampled_before_learning_starts"] = 2000
    # alg_config["train_batch_size"] = 256
    #
    # # 4) Target Network Updates & Gradient Clipping
    # alg_config["tau"] = 0.0001
    # alg_config["target_network_update_freq"] = 4
    # alg_config["grad_clip"] = 5.0
    #
    # # 5) Exploration: Zero Noise Well Before the End
    # alg_config["exploration_config"] = {
    #     "type": "OrnsteinUhlenbeckNoise",
    #     "random_timesteps": 2000,
    #     "initial_scale": 1.0,
    #     "final_scale": 0.0,  # Absolutely no noise in the final phase
    #     "scale_timesteps": 50000,  # Zero noise by step 50k
    #     "ou_base_scale": 0.1,
    #     "ou_theta": 0.15,
    #     "ou_sigma": 0.2,
    # }
    # alg_config.twin_q = False

    # ---------------------------------------------------------------------
    alg = alg_config.build()

    n_training_episodes *= env_config['num_Haar_basis'] * env_config['steps_per_Haar']

    update_every_percent = 2
    results = []
    update_interval = max(1, int(n_training_episodes * (update_every_percent / 100)))

    training_start_time = get_time()
    # ---------------------> Train Agent <-------------------------
    for i in range(n_training_episodes):
        results.append(alg.train())
        # Print update every x%
        if (i + 1) % int(update_interval) == 0 or (i + 1) == n_training_episodes:
            percent_complete = (i + 1) / n_training_episodes * 100
            print(f"Training Progress: {percent_complete:.0f}% complete")

    training_end_time = get_time()
    training_elapsed_time = training_end_time - training_start_time


    save_dir = ""

    # ---------------------> Save Results <-------------------------
    if save is True:
        env = alg.workers.local_worker().env
        sr = SaveResults(env, alg)
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    config_table(env_config=env_config,alg_config=alg_config,filepath=save_dir)

    # ---------------------> Plot Data <-------------------------
    if plot is True:
        assert save is True, "If plot=True, then save must also be set to True"
        env_string = "Noisy " if isinstance(env, NoisyChangingTargetEnv) else "Noiseless"
        training_figure_title = " ".join(f"{target_gate}-" for target_gate in env_config["U_target_list"])
        plot_data(save_dir = save_dir, figure_title=env_string + training_figure_title, plot_filename='Training')
        print("Plots Created")
    # --------------------------------------------------------------

    return alg, training_elapsed_time, save_dir

def inference_and_save(inference_list, save_dir, train_alg, n_episodes_for_inferencing):
    columns = ['Fidelity', 'Rewards', 'Actions', 'Self.U_Operator', 'Target_Operator', 'Original_target', 'Original_U_initial', 'Episode Id']


    for curr_gate in inference_list:
        # train_alg = copy.deepcopy(alg)

        gate_save_dir = save_dir + f'/{curr_gate}/'
        plot_filename = f'inference_{curr_gate}.png'
        os.makedirs(gate_save_dir)

        env_string = ""


        env_data_title = f"{curr_gate}_"
        transition_history = []

        for inference_iteration in range(n_episodes_for_inferencing):
            # -----------------------> Inferencing <---------------------------
            env = train_alg.workers.local_worker().env
            inference_env, target_gate, history = do_inferencing(env, train_alg, curr_gate)
            transition_history.append(history)
            env_string = "Noisy " if isinstance(env, NoisyChangingTargetEnv) else "Noiseless"

        figure_title = f"[{env_string}] Inferencing on Multiple Different {str(curr_gate)}."
        df = pd.DataFrame(transition_history, columns=columns)
        # df.to_pickle(env_data_title + "env_data.pkl")  # easier to load than csv
        df.to_csv(gate_save_dir + env_data_title + "env_data.csv", index=False)  # backup in case pickle doesn't work
        multiple_inference_visuals(df, figure_title=figure_title, save_dir=gate_save_dir, plot_filename=plot_filename,
                               gate=curr_gate)




def do_inferencing(env, train_alg, curr_gate):
    """
    alg: The trained model
    n_episodes_for_inferencing: Number of episodes to do during the training
    """

    # Initialize a new environment for inference using this configuration
    inference_env_config = env.return_env_config()
    target_gate = curr_gate.get_matrix()  # Set new target gate for inference

    inference_env_config["U_target_list"] = [curr_gate]
    inference_env_config['verbose'] = False
    inference_env_config["U_target"] = target_gate
    env_class = type(env)
    inference_env = env_class(inference_env_config)

    # ------------------------------------------------------------------------------------
    target_gate = np.array(target_gate)

    episode_reward = 0.0
    done = False

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
    env = NoisyChangingTargetEnv
    # env = ChangingTargetEnv
    n_training_episodes = 60
    save = True
    plot = True
    n_episodes_for_inferencing = 1000
    u_target_list = [gates.RandomSU2()]
    alg, training_time, save_dir = run(env, n_training_episodes, u_target_list, save, plot)
    inferencing_gate = [gates.RandomSU2(), gates.Rx(), gates.Ry(), gates.Rz(),
                        gates.X(), gates.Y(), gates.Z(), gates.H(), gates.S(),gates.XY_combination(),gates.ZX_combination(),gates.HS()]

    inference_start = get_time()
    inference_and_save(inference_list=inferencing_gate, save_dir=save_dir, train_alg=alg,
                       n_episodes_for_inferencing=n_episodes_for_inferencing)
    inference_end = get_time()
    inference_elapsed_time = inference_end - inference_start
    print(f"Training Time: {training_time}")
    print(f"Inference Time + Saving Inference + Inference Visuals: {inference_elapsed_time}")
    print(f'Results saved to: {save_dir}')

if __name__ == "__main__":
    main()
