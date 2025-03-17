import random
import numpy as np
from relaqs.environments import SingleQubitEnv, NoisySingleQubitEnv
from relaqs.api.gates import RandomSU2
from relaqs.api.utils import normalize

class ChangingTargetEnv(SingleQubitEnv):
    @classmethod
    def get_default_env_config(cls):
        config_dict = super().get_default_env_config()
        # config_dict["observation_space_size"] = 17
        # config_dict["observation_space_size"] = 25
        config_dict["observation_space_size"] = 8
        config_dict["U_target_list"] = []
        config_dict["target_generation_function"] = RandomSU2
        return config_dict
    
    def __init__(self, env_config):
        super().__init__(env_config)
        self.U_target_list = env_config["U_target_list"]
        self.target_generation_function = env_config["target_generation_function"]

    def set_target_gate(self):
        if len(self.U_target_list) == 0:
            self.U_target= self.target_generation_function().get_matrix()
        else:
            self.U_target = random.choice(self.U_target_list).get_matrix()
        self.original_U_target = self.U_target

    def set_initial_gate(self):
        # if len(self.U_target_list) == 0:
        #     self.U_initial= self.target_generation_function().get_matrix()
        # else:
        #     self.U_initial = random.choice(self.U_target_list).get_matrix()
        self.U_initial = self.target_generation_function().get_matrix()
        self.original_U_initial = self.U_initial.copy()

    def reset(self, *, seed=None, options=None):
        _, info = super().reset()
        self.set_target_gate()
        self.set_initial_gate()
        starting_observation = self.get_observation()
        return starting_observation, info

    def get_observation(self):
        # observation = super().get_observation()
        # U_diff1 = self.U_target @ self.U.conj().T
        U_diff = self.U_target @ self.U_initial.conj().T
        # return np.append([self.compute_fidelity()], self.unitary_to_observation(U_diff2))
        return self.unitary_to_observation(U_diff)
        # return np.append(self.unitary_to_observation(U_diff1),self.unitary_to_observation(U_diff2))
        # temp = np.append(observation, self.unitary_to_observation(self.U_target))
        # return np.append(temp, self.unitary_to_observation(self.U_initial))
        # return np.append(observation, self.unitary_to_observation(self.U_target))

    def return_env_config(self):
        env_config = super().get_default_env_config()
        env_config.update({
                           "observation_space_size": 8,
                           "num_Haar_basis": self.num_Haar_basis,
                           "steps_per_Haar": self.steps_per_Haar,
                           "verbose": self.verbose,
                           "U_init": self.U_initial,
                           "U_target": self.U_target,
                           "target_generation_function": self.target_generation_function,
                           "U_target_list": self.U_target_list,
                           })
        return env_config
    
class NoisyChangingTargetEnv(ChangingTargetEnv, NoisySingleQubitEnv):
    @classmethod
    def get_default_env_config(cls):
        config_dict = super().get_default_env_config()
        config_dict["observation_space_size"] = 16  # Reduced observation space
        config_dict["reward_shaping"] = True
        config_dict["shaping_weight"] = 2.0  # Weight for shaping component
        return config_dict
    
    def __init__(self, env_config):
        super().__init__(env_config)
        self.U_target_list = env_config["U_target_list"]
        self.target_generation_function = env_config["target_generation_function"]
        self.reward_shaping = env_config.get("reward_shaping", True)
        self.shaping_weight = env_config.get("shaping_weight", 2.0)
        self.prev_fidelity = 0.0
        self.prev_diff_norm = None
        
    def set_target_gate(self):
        if len(self.U_target_list) == 0:
            U = self.target_generation_function().get_matrix()
        else:
            U = random.choice(self.U_target_list).get_matrix()
        self.U_target = self.unitary_to_superoperator(U)
        self.original_U_target = U.copy()

    def set_initial_gate(self):
        self.original_U_initial = self.target_generation_function().get_matrix()
        self.U_initial = self.unitary_to_superoperator(self.original_U_initial.copy())

    def compute_gate_difference(self):
        """Compute normalized difference between target and current gate"""
        U_diff = self.U_target @ self.U.conj().T  # Use current U, not U_initial
        diff_norm = np.linalg.norm(U_diff - np.eye(4)) / 4.0
        return diff_norm, U_diff

    def extract_essential_components(self, matrix):
        """Extract the most informative components from a matrix"""
        # Flatten the matrix and find largest magnitude elements
        flat = matrix.flatten()
        real_indices = np.argsort(np.abs(flat.real))[-6:]  # 6 largest real components
        imag_indices = np.argsort(np.abs(flat.imag))[-6:]  # 6 largest imaginary components
        
        # Get unique indices
        indices = np.unique(np.concatenate([real_indices, imag_indices]))[:12]
        
        # Extract values
        essential = np.zeros(12, dtype=np.float32)
        for i, idx in enumerate(indices[:12]):
            if i < len(indices):
                val = flat[idx]
                essential[i] = val.real if i % 2 == 0 else val.imag
                
        return essential

    def get_observation(self):
        """Optimized observation space focusing on most informative components"""
        # Get current state information
        fidelity = self.compute_fidelity()
        diff_norm, U_diff = self.compute_gate_difference()
        
        # Extract essential components instead of using full matrix
        essential_components = self.extract_essential_components(U_diff)
        
        # Noise parameters (normalized)
        normalized_t1 = normalize(self.relaxation_rate[0], self.relaxation_rates_list[0])
        normalized_t2 = normalize(self.relaxation_rate[1], self.relaxation_rates_list[1])
        normalized_detuning = normalize(self.detuning, self.detuning_list)
        
        # Combine the most important information
        # Total: 16 dimensions (more focused)
        obs = np.concatenate([
            [fidelity],                # Current fidelity (1)
            [diff_norm],               # Distance metric (1)
            essential_components,      # Essential matrix components (12)
            [normalized_t1, normalized_t2, normalized_detuning]  # Noise parameters (3)
        ])
        return obs

    def compute_reward(self, fidelity):
        """Enhanced reward function with focused shaping"""
        # Base reward (typically based on fidelity)
        base_reward = super(ChangingTargetEnv, self).compute_reward(fidelity)
        
        if not self.reward_shaping:
            return base_reward
            
        # Reward for fidelity improvement
        fidelity_improvement = fidelity - self.prev_fidelity
        
        # Reward for getting closer to target
        diff_norm, _ = self.compute_gate_difference()
        if self.prev_diff_norm is not None:
            diff_improvement = self.prev_diff_norm - diff_norm
        else:
            diff_improvement = 0
            
        # Update tracking variables
        self.prev_diff_norm = diff_norm
        
        # Calculate shaped reward
        # More emphasis on absolute fidelity, but also reward progress
        shaped_reward = base_reward + self.shaping_weight * (fidelity_improvement + diff_improvement)
        
        return shaped_reward

    def reset(self, *, seed=None, options=None):
        self.prev_diff_norm = None
        self.prev_fidelity = 0.0
        obs, info = super().reset()
        self.relaxation_rate = self.get_relaxation_rate()
        self.detuning_update()
        return self.get_observation(), info

    def return_env_config(self):
        env_config = super().return_env_config()
        env_config.update({
            "detuning_list": self.detuning_list,
            "relaxation_rates_list": self.relaxation_rates_list,
            "relaxation_ops": self.relaxation_ops,
            "observation_space_size": 16,
            "reward_shaping": self.reward_shaping,
            "shaping_weight": self.shaping_weight
        })
        return env_config


