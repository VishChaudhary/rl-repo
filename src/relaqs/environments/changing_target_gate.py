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
        config_dict["observation_space_size"] = 35
        return config_dict
    
    def __init__(self, env_config):
        super().__init__(env_config)
        self.U_target_list = env_config["U_target_list"]
        self.target_generation_function = env_config["target_generation_function"]

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

    def get_observation(self):
        normalized_detuning = [normalize(self.detuning, self.detuning_list)]
        normalized_relaxation_rates = [normalize(self.relaxation_rate[0], self.relaxation_rates_list[0]),
                                       normalize(self.relaxation_rate[1],
                                                 self.relaxation_rates_list[1])]  # could do list comprehension

        U_diff = self.U_target @ self.U_initial.conj().T
        obs_diff = self.unitary_to_observation(U_diff)
        # return obs_diff
        return np.append(normalized_relaxation_rates + normalized_detuning, obs_diff)

    def return_env_config(self):
        env_config = super().return_env_config()
        env_config.update({"detuning_list": self.detuning_list,  # qubit detuning
                           "relaxation_rates_list": self.relaxation_rates_list,
                           "relaxation_ops": self.relaxation_ops,
                           "observation_space_size": 35,
                           })
        return env_config


