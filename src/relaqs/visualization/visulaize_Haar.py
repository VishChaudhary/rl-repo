from qutip.qobj import *
from qutip.bloch import *
import qutip
from qutip import sigmax, sigmaz, sigmay, qeye
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from relaqs import RESULTS_DIR

# def plot_bloch_sphere_state(state):
#     """ State can be a state vector or density matrix """
#     fig = plt.figure()
#     b = qutip.Bloch(fig=fig)
#     b.add_states(qutip.Qobj(state))  # need to convert to Qobj
#     b.render()
#     plt.show()


def load_and_plot(file_path, window_size):
    df = pd.read_csv(RESULTS_DIR + file_path)

    size = len(df['gamma_magnitude'])
    # Extract the pulse parameters
    gamma_magnitude = df['gamma_magnitude'].tail(window_size)
    gamma_phase = df['gamma_phase'].tail(window_size)
    alpha = df['alpha'].tail(window_size)
    current_Haar_num = df['current_Haar_num'].tail(window_size)
    current_step_per_Haar = df['current_step_per_Haar'].tail(window_size)
    fidelity = df['fidelity'].tail(window_size)

    # Create a time vector based on the number of control pulses
    start = size - window_size
    end = size
    time_steps = np.arange(start, end, 1)
    # time_steps = 8

    # Create the plots
    fig, axs = plt.subplots(6, 1, figsize=(10, 8))

    # Plot gamma_magnitude
    axs[0].plot(time_steps, gamma_magnitude, label='gamma_magnitude', color='b')
    axs[0].set_title('Gamma Magnitude Over Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)

    # Plot gamma_phase
    axs[1].plot(time_steps, gamma_phase, label='gamma_phase', color='g')
    axs[1].set_title('Gamma Phase Over Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Phase (radians)')
    axs[1].grid(True)

    # Plot alpha (detuning)
    axs[2].plot(time_steps, alpha, label='alpha', color='r')
    axs[2].set_title('Alpha (Detuning) Over Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Detuning (Hz)')
    axs[2].grid(True)

    # Plot current Haar num
    axs[3].plot(time_steps, current_Haar_num, label='current_Haar_num', color='black')
    axs[3].set_title('Haar Basis Over Time')
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('#')
    axs[3].grid(True)

    # Plot current Step per Haar
    axs[4].plot(time_steps, current_step_per_Haar, label='current_step_per_Haar', color='cyan')
    axs[4].set_title('Step Per Haar')
    axs[4].set_xlabel('Time Step')
    axs[4].set_ylabel('#')
    axs[4].grid(True)

    # Plot Fidelity
    axs[5].plot(time_steps, fidelity, label='fidelity', color='green')
    axs[5].set_title('Fidelity Over Time')
    axs[5].set_xlabel('Time Step')
    axs[5].set_ylabel('Fidelity')
    axs[5].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show()


if __name__ == "__main__":
    load_and_plot('2024-10-20_15-52-50_Y/pulse_data.csv', 64)


