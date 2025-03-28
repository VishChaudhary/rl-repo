""" Single qubit gate synthesis environment using Haar basis. """
import cmath
import gymnasium as gym
import numpy as np
import scipy.linalg as la

from relaqs.api import gates

I = gates.I().get_matrix()
X = gates.X().get_matrix()
Y = gates.Y().get_matrix()
Z = gates.Z().get_matrix()

class SingleQubitEnv(gym.Env):
    @classmethod
    def get_default_env_config(cls):
        return {
            "action_space_size": 3,
            "U_initial": I,
            "U_target": X,  
            "final_time": 35.5556E-9, # in seconds
            "num_Haar_basis": 1,
            "steps_per_Haar": 2,  # steps per Haar basis per episode
            "verbose": True,
            "observation_space_size": 9,  # 1 (fidelity) + 8 (flattened unitary)
        }
    def __init__(self, env_config):
        self.final_time = env_config["final_time"]  # Final time for the gates
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(env_config["observation_space_size"],))
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]))
        self.U_target = env_config["U_target"]
        self.U_target_dm = None
        self.U_initial_dm = None
        self.U_initial = env_config["U_initial"] # future todo, can make random initial state
        self.U = env_config["U_initial"].copy()
        self.num_Haar_basis = env_config["num_Haar_basis"]
        self.steps_per_Haar = env_config["steps_per_Haar"]
        self.verbose = env_config["verbose"]
        self.current_Haar_num = 1
        self.current_step_per_Haar = 1
        self.H_array = []
        self.H_tot = []
        self.U_array = []
        self.state = self.unitary_to_observation(self.U)
        self.prev_fidelity = 0
        self.gamma_phase_max = 1.1675 * np.pi
        self.gamma_magnitude_max = 1.8 * np.pi / self.final_time / self.steps_per_Haar
        self.alpha_max = 0.05E9 # detuning of the control pulse in Hz
        self.transition_history = []
        self.episode_id = 0

    def unitary_to_observation(self, U):
        return (
            np.array(
                [(abs(x), (cmath.phase(x) / np.pi + 1) / 2) for x in U.flatten()],
                dtype=np.float64,
            )
            .squeeze()
            .reshape(-1)
        )

    def get_observation(self):
        return np.append([self.compute_fidelity()], self.unitary_to_observation(self.U))
    
    def compute_fidelity(self):
        U_target_dagger = self.U_target.conjugate().transpose()
        return float(np.abs(np.trace(U_target_dagger @ self.U))) / (self.U.shape[0])
    
    def compute_reward(self, fidelity):
        return (-3 * np.log10(1.0 - fidelity) + np.log10(1.0 - self.prev_fidelity)) + (3 * fidelity - self.prev_fidelity)
        
    def hamiltonian(self, alpha, gamma_magnitude, gamma_phase):
        return alpha*Z + gamma_magnitude*(np.cos(gamma_phase)*X + np.sin(gamma_phase)*Y)

    def reset(self, *, seed=None, options=None):
        self.U = self.U_initial.copy()
        starting_observeration = self.get_observation()
        self.state = self.get_observation()
        self.current_Haar_num = 1
        self.current_step_per_Haar = 1
        self.H_array = []
        self.H_tot = []
        self.U_array = []
        self.prev_fidelity = 0
        info = {}
        self.episode_id += 1
        if self.verbose is True:
            print("episode id: ", self.episode_id)
        return starting_observeration, info
    
    def hamiltonian_update(self, num_time_bins, *hamiltonian_args):
        H = self.hamiltonian(*hamiltonian_args)
        self.H_array.append(H)
        self.H_tot = []
        for ii, H_elem in enumerate(self.H_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(ii / self.steps_per_Haar) # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1)))
                if ii > 0:
                    self.H_tot[jj] += factor * H_elem
                else:
                    self.H_tot.append(factor * H_elem)

    def is_episode_over(self, fidelity):
        truncated = False
        terminated = False
        if fidelity >= 1:
            truncated = True  # truncated when target fidelity reached
        elif (self.current_Haar_num >= self.num_Haar_basis) and (self.current_step_per_Haar >= self.steps_per_Haar):  # terminate when all Haar is tested
            terminated = True
        return truncated, terminated

    def Haar_update(self):
        if (self.current_step_per_Haar == self.steps_per_Haar):  # For each Haar basis, if all trial steps ends, them move to next haar wavelet
            self.current_Haar_num += 1
            self.current_step_per_Haar = 1
        else:
            self.current_step_per_Haar += 1

    def parse_actions(self, action):
        gamma_magnitude = self.gamma_magnitude_max / 2  * (action[0] + 1)
        gamma_phase = self.gamma_phase_max * action[1]
        alpha = self.alpha_max * action[2]
        return gamma_magnitude, gamma_phase, alpha
    
    def update_transition_history(self, fidelity, reward, action):
        self.transition_history.append([fidelity, reward, action, self.U, self.U_target,  self.U_target_dm, self.U_initial_dm, self.episode_id])

    def get_info(self, fidelity, reward, action, truncated, terminated):
        info_string = f"""Step: {self.current_step_per_Haar}
            F: {fidelity:7.3f}
            R: {reward:7.3f}
            amp: {action[0]:7.3f}
            phase: {action[1]:7.3f}
            alpha: {action[2]:7.3f}
            truncated: {truncated}
            terminated: {terminated}
            """
        return info_string

    def step(self, action):
        num_time_bins = 2 ** (self.current_Haar_num - 1) # Haar number decides the number of time bins

        # Get actions
        gamma_magnitude, gamma_phase, alpha = self.parse_actions(action)

        self.hamiltonian_update(num_time_bins, alpha, gamma_magnitude, gamma_phase)

        # U update
        self.U = self.U_initial.copy()
        for jj in range(0, num_time_bins):
            Ut = la.expm(-1j * self.final_time / num_time_bins * self.H_tot[jj])
            self.U = Ut @ self.U
        self.U_array.append(self.U)

        # Get reward (fidelity)
        fidelity = self.compute_fidelity()
        reward = self.compute_reward(fidelity)
        self.prev_fidelity = fidelity

        self.state = self.get_observation()

        self.update_transition_history(fidelity, reward, action)

        truncated, terminated = self.is_episode_over(fidelity)

        if self.verbose is True:
            print(self.get_info(fidelity, reward, action, truncated, terminated))
            
        self.Haar_update()

        info = {}
        return (self.state, reward, terminated, truncated, info)
