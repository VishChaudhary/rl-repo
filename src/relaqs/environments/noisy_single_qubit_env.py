""" Noisy single qubit gate synthesis environment using Haar basis. """
import random
import numpy as np
import scipy.linalg as la
from qutip import Qobj
from qutip.superoperator import liouvillian, spre, spost
from qutip.operators import sigmam, sigmaz
from relaqs.environments.single_qubit_env import SingleQubitEnv
from relaqs.api import gates
from relaqs.api.utils import sample_noise_parameters, normalize
import cmath

I = gates.I().get_matrix()
X = gates.X().get_matrix()
Y = gates.Y().get_matrix()
Z = gates.Z().get_matrix()

class NoisySingleQubitEnv(SingleQubitEnv):
    @classmethod
    def get_default_env_config(cls):
        env_config = super().get_default_env_config()
        t1_list, t2_list, detuning_list = sample_noise_parameters()
        env_config.update({"detuning_list": detuning_list,  # qubit detuning
                           "fidelity_threshold": 0.8,
                           "fidelity_target_switch_case": 20,
                           "base_target_switch_case": 1000,
                           "relaxation_rates_list": [t1_list, t2_list],
                           # relaxation lists of list of floats to be sampled from when resetting environment. (10 usec)
                           "relaxation_ops": [sigmam(), sigmaz()],
                           # relaxation operator lists for T1 and T2, respectively
                           "observation_space_size": 2 * 16 + 1 + 2 + 1 + 32})  # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rates + 1 for detuning})
        return env_config


    def __init__(self, env_config):
        super().__init__(env_config)
        self.detuning_list = env_config["detuning_list"]
        self.detuning_update()
        self.original_U_target = env_config["U_target"]
        self.U_target = self.unitary_to_superoperator(env_config["U_target"])
        self.U_initial = self.unitary_to_superoperator(env_config["U_initial"])
        self.episode_switched = []
        self.global_episode_num = 0
        self.local_episode_num = 0
        self.fidelity_threshold = env_config["fidelity_threshold"]
        self.continuous_fidelity_count = [0] * self.num_Haar_basis * self.steps_per_Haar
        self.fidelity_target_switch_case = env_config["fidelity_target_switch_case"]
        self.base_target_switch_case = env_config["base_target_switch_case"]
        self.relaxation_rates_list = env_config["relaxation_rates_list"]
        self.relaxation_ops = env_config["relaxation_ops"]
        self.relaxation_rate = self.get_relaxation_rate()
        self.U = self.U_initial.copy()  # multiplied propagtion operators
        self.state = self.unitary_to_observation(self.U_initial)  # starting observation space

    def return_env_config(self):
        env_config = super().get_default_env_config()
        env_config.update({"detuning_list": self.detuning_list,  # qubit detuning
                           "fidelity_threshold": 0.8,
                           "fidelity_target_switch_case": 20,
                           "base_target_switch_case": 1000,
                           #            "relaxation_rates_list": [[0.01,0.02],[0.05, 0.07]], # relaxation lists of list of floats to be sampled from when resetting environment.
                           #            "relaxation_ops": [sigmam(),sigmaz()] #relaxation operator lists for T1 and T2, respectively  # qubit detuning
                           "relaxation_rates_list": self.relaxation_rates_list, # relaxation lists of list of floats to be sampled from when resetting environment. (10 usec)
                           "relaxation_ops": self.relaxation_ops, # relaxation operator lists for T1 and T2, respectively
                           "observation_space_size": 68,
                           "num_Haar_basis": self.num_Haar_basis,
                           "steps_per_Haar": self.steps_per_Haar,
                           })
        return env_config

    def detuning_update(self):
        # Random detuning selection
        if len(self.detuning_list) == 1:
            self.detuning = self.detuning_list[0]
        else:
            self.detuning = random.sample(self.detuning_list, k=1)[0]

    @classmethod
    def unitary_to_superoperator(self, U):
        # return np.kron(U.conj(), U)
        return (spre(Qobj(U)) * spost(Qobj(U))).data.toarray()

    def get_relaxation_rate(self):
        relaxation_size = len(self.relaxation_ops) # get number of relaxation ops
        
        sampled_rate_list = []
        for ii in range(relaxation_size):
            sampled_rate_list.append(random.sample(self.relaxation_rates_list[ii],k=1)[0])

        return sampled_rate_list

    def get_observation(self):
        normalized_detuning = [normalize(self.detuning, self.detuning_list)]
        normalized_relaxation_rates = [normalize(self.relaxation_rate[0], self.relaxation_rates_list[0]),
                                       normalize(self.relaxation_rate[1], self.relaxation_rates_list[1])] # could do list comprehension

        interm_array = np.append([self.compute_fidelity()] +
                         normalized_relaxation_rates +
                         normalized_detuning,
                         self.unitary_to_observation(self.U))

        return np.append(interm_array, self.unitary_to_observation(self.U_target))

    def hamiltonian(self, detuning, alpha, gamma_magnitude, gamma_phase):
        return (detuning + alpha)*Z + gamma_magnitude*(np.cos(gamma_phase)*X + np.sin(gamma_phase)*Y)

    def hamiltonian_update(self, num_time_bins, detuning, alpha, gamma_magnitude, gamma_phase):
        H = self.hamiltonian(detuning, alpha, gamma_magnitude, gamma_phase)
        self.H_array.append(H)
        self.H_tot = []
        for ii, H_elem in enumerate(self.H_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(
                    ii / self.steps_per_Haar)  # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1)))
                if ii > 0:
                    self.H_tot[jj] += factor * H_elem
                else:
                    self.H_tot.append(factor * H_elem)

    def reset(self, *, seed=None, options=None):
        super().reset()
        self.state = self.get_observation()
        self.relaxation_rate = self.get_relaxation_rate()
        self.detuning_update()
        starting_observeration = self.get_observation()
        info = {}
        return starting_observeration, info
    
    def operator_update(self, num_time_bins):
        # Set noise opertors
        jump_ops = []
        for ii in range(len(self.relaxation_ops)):
            jump_ops.append(np.sqrt(self.relaxation_rate[ii]) * self.relaxation_ops[ii])

        self.U = self.U_initial.copy()
        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot[jj]), jump_ops, data_only=False, chi=None)).data.toarray()  # Liouvillian calc
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at

    def get_info(self, fidelity, reward, action, truncated, terminated):
        info_string = super().get_info(fidelity, reward, action, truncated, terminated)
        info_string += f"""Relaxation rate: {self.relaxation_rate}
            Detuning: {self.detuning}"""
        return info_string

    def get_self_U(self):
        return self.U

    def get_self_episode_num(self):
        return self.global_episode_num

    def get_episodes_gate_switch(self):
        return self.episode_switched

    def switch_target_gate(self, fidelity):
        self.global_episode_num += 1
        self.local_episode_num += 1
        fidelity_threshold_reached = False
        base_num_episodes_trained = False
        haar_val = self.current_Haar_num * self.current_step_per_Haar
        fidelity_count_idx = haar_val - 1
        num_vals_needed = (haar_val + 1) // 2
        vals_above_fidelity_count = 0

        if fidelity >= self.fidelity_threshold:
            self.continuous_fidelity_count[fidelity_count_idx] += 1

        else:
            self.continuous_fidelity_count[fidelity_count_idx] = 0

        # vals_above_fidelity_count = sum(val >= self.fidelity_target_switch_case for val in self.continuous_fidelity_count)
        for val in self.continuous_fidelity_count:
            if val >= self.fidelity_target_switch_case:
                vals_above_fidelity_count += 1

        if vals_above_fidelity_count >= num_vals_needed:
            fidelity_threshold_reached = True

        if self.local_episode_num >= self.base_target_switch_case:
            base_num_episodes_trained = True

        #Switch Target Gate
        if fidelity_threshold_reached and base_num_episodes_trained:
            self.local_episode_num = 0
            self.continuous_fidelity_count = [0] * self.num_Haar_basis * self.steps_per_Haar

            random_gate = gates.RandomSU2()
            self.U_target = self.unitary_to_superoperator(random_gate.get_matrix())
            self.episode_switched.append(self.global_episode_num)

            print(
                "\n----------------------------------------------------------------------------U_TARGET-------------------------------------------------------------------------------------------------------------\n")
            print(self.U_target)
            print(
                "\n-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n")


    ###Obviously update this
    def step(self, action, training = True):
        num_time_bins = 2 ** (self.current_Haar_num - 1) # Haar number decides the number of time bins

        # gamma is the complex amplitude of the control field
        gamma_magnitude, gamma_phase, alpha = self.parse_actions(action)

        ##Create this in this file
        self.hamiltonian_update(num_time_bins, self.detuning, alpha, gamma_magnitude, gamma_phase)

        self.operator_update(num_time_bins)

        # Reward and fidelity calculation
        fidelity = self.compute_fidelity()
        reward = self.compute_reward(fidelity)
        self.prev_fidelity = fidelity

        self.state = self.get_observation()

        if training:
            self.switch_target_gate(fidelity)

        self.update_transition_history(fidelity, reward, action)

        truncated, terminated = self.is_episode_over(fidelity)

        if self.verbose is True:
            print(f'------------Step call: {self.global_episode_num}-----------------------')
            print(self.get_info(fidelity, reward, action, truncated, terminated))


        self.Haar_update()

        info = {}
        return (self.state, reward, terminated, truncated, info)

if __name__ == "__main__":
    print("hello")
