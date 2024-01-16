import numpy as np
import matplotlib.pyplot as plt

import os
import logging

import itertools
from typing import Callable

from tqdm import tqdm
import time

"""
k_B = 1.0 for the whole script

SCRIPT HAS TO BE CHECKED, BUT SHOULD BE ABLE TO RUN LIKE THIS
"""

def compute_energy(configuration: np.ndarray, l_box: np.ndarray) -> float:
    """
    Computes the Energy of a two dimensional Ising configuration

    Args:
        configuration (np.ndarray):
            Numpy array containing the information of the spin orientation of an 2D Ising model
    
    Returns:
        energy (float): Energy of the configuration
    """
    configuration = configuration.reshape(tuple(l_box))
    L1, L2 = l_box
    energy = 0.0
    
    for i_indx in range(L1):
        nearest_neighbors_locations_i_dim = [(i_indx - 1) % L1, (i_indx + 1) % L1]
        for j_indx in range(L2):
            nearest_neighbors_locations_j_dim = [(j_indx - 1) % L2, (j_indx + 1) % L2]
            spin = configuration[i_indx, j_indx]

            nearest_neighbors = np.concatenate(
                (configuration[nearest_neighbors_locations_i_dim, j_indx], configuration[i_indx, nearest_neighbors_locations_j_dim])
            )

            energy -= spin * np.sum(nearest_neighbors) 
    return 0.5 * energy


def compute_magnetization(configuration: np.ndarray, l_box: np.ndarray) -> float: 
    return np.abs(np.sum(configuration))


def boltzmann_probability(temp: float, l_box: np.ndarray, configuration: np.ndarray) -> float:
    energy = compute_energy(configuration, l_box)
    beta = 1/temp
    return np.exp(-1 * beta * energy)


def state_space(l_box: np.ndarray) -> np.ndarray:
    """
    Returns all possible spin configurations dependent on the dimensions of the spin system
    
    Args: 
        l_box (np.ndarray):
            array of spin sites per dimension

    Returns:
        state_space (np.ndarray):
            One dimensional array containing all spin configurations (as Numpy arrays)
    """
    state_space = itertools.product([-1, 1], repeat=np.prod(l_box))
    return np.array(list(state_space))

# Finish helper
def compute_mean_observable(compute_observable_func: Callable, configuration: np.ndarray, l_box: np.ndarray, boltzmann_proba: float) -> float:
    """
    Computes the mean value of an observable in an 2D Ising model.

    Args:
        compute_observable_func (Callable):
            Function that computes the value of the observable dependent on the spin-configuration
    """
    observable_value = compute_observable_func(configuration, l_box)
    
    return boltzmann_proba * observable_value


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)

    # testing if the partition function is the same as the computation in worksheet2
    l_box = np.array([4, 4])
    temp_list = np.arange(1.0, 5.1, 0.1).round(2)

    logging.info('Computations started')

    start_time = time.time()

    total_state_space = state_space(l_box)

    energy_list = []
    magnetization_list = []
    for temperature in tqdm(temp_list):
        partition_function = 0.0
        energy_per_site = 0.0
        magnetization_per_site = 0.0
        for configuration in tqdm(total_state_space):
            boltzmann_proba = boltzmann_probability(temperature, l_box, configuration)

            partition_function += boltzmann_proba
            energy_per_site += compute_mean_observable(
                compute_energy,
                configuration, 
                l_box,
                boltzmann_proba,
            ) 
            magnetization_per_site += compute_mean_observable(
                compute_magnetization,
                configuration, 
                l_box,
                boltzmann_proba,
            )
        energy_per_site /= (partition_function * np.prod(l_box))
        magnetization_per_site /= (partition_function * np.prod(l_box))

        energy_list.append(energy_per_site)
        magnetization_list.append(magnetization_per_site)

    duration = (time.time() - start_time) / 60.0 # duration in Minutes
    logging.info('Exact Summation computations finished.')

    save_array = np.array(
        [temp_list, energy_list, magnetization_list]
    )
    np.save('./sm1_worksheet_5/plots/Ising_exact.npy', save_array)

    plt.plot(temp_list, energy_list, label=r'energy per site $e$')
    plt.plot(temp_list, magnetization_list, label=r'magnetization per site $m$')
    plt.legend(loc='upper right')
    plt.xlabel('temperature')
    plt.ylabel('mean observable [a.u.]')
    plt.title(f'Duration: {round(duration, 2)} min')
    
    # plt.savefig(f'./sm1_worksheet_5/plots/exact_Ising_observables.png', format='png', dpi=150)
    plt.show()
    
    