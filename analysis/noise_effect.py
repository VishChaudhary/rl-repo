"""
zero_noise = True verifies that the noisy environment matches the noiseless environment when there
no noise parameters are provided.

zero_noise = False samples T1 and T2 data from IBM and samples detuning from a normal distribution.
This setting verifies that the noise has a significant effect on the fidelity.
"""
import numpy as np
from relaqs.api.utils import sample_noise_parameters
from relaqs.environments import SingleQubitEnv, NoisySingleQubitEnv
from relaqs.api.gates import RandomSU2
from qutip.operators import sigmaz, sigmam

zero_noise = False

# Define environments
noiseless_env = SingleQubitEnv(SingleQubitEnv.get_default_env_config())
env_config = NoisySingleQubitEnv.get_default_env_config()
if zero_noise:
    env_config["relaxation_rates_list"] = []
    env_config["relaxation_ops"] = []
    env_config["detuning_list"] = [0]
else: # sample noise
    # Test with IBM T1 and T2 data and sampling detuning from a normal distribution
    noise_file = "april/ibmq_belem_month_is_4.json"
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)

    env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()]
    env_config["detuning_list"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(), sigmaz()]
noisy_env = NoisySingleQubitEnv(env_config)

# Set random U_target
U_target = RandomSU2().get_matrix()
noiseless_env.U_target = U_target
noisy_env.U_target = NoisySingleQubitEnv.unitary_to_superoperator(U_target)

# Take random actions
actions = noiseless_env.action_space.sample()
noiseless_env.step(actions)
noisy_env.step(actions)

# Get superoperators
noiseless_superop = NoisySingleQubitEnv.unitary_to_superoperator(noiseless_env.U)
noisy_superop = noisy_env.U

print("super op close after step:", np.allclose(noiseless_superop, noisy_superop))

# Compute superoperator fidelities
noisy_fidelity = noisy_env.compute_fidelity()
noisy_env.U = noiseless_superop
noiseless_fidelity = noisy_env.compute_fidelity()

# Compute superoperator fidelity difference
superop_fidelity_difference = noiseless_fidelity - noisy_fidelity
print("Superoperator fidelitiy difference:", superop_fidelity_difference)
