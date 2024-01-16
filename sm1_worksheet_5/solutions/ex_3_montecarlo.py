import numpy as np
import matplotlib.pyplot as plt

import os
import logging

import itertools
from typing import Callable, Tuple

from tqdm import tqdm
import functools

from ex_3_exact import compute_energy, compute_magnetization, compute_mean_observable, state_space, boltzmann_probability

def ising_trial_move(configuration: np.ndarray) -> np.ndarray:
    number_of_spin = len(configuration)
    trial_state = configuration.copy()
    trial_state[np.random.randint(0, number_of_spin)] *= -1
    
    return trial_state 

def evaluate_observables(configuration_dynamics: np.ndarray, l_box: np.ndarray):
    """
    write helper
    """
    energy_list = [compute_energy(config, l_box) for config in configuration_dynamics]
    magnetization_list = [compute_magnetization(config, l_box) for config in configuration_dynamics]

    return np.mean(energy_list)/np.prod(l_box), np.mean(magnetization_list)/np.prod(l_box)

def metropolis(
        N: int,
        P: Callable, 
        trial_move: Callable, 
        initial_config: np.ndarray,
) -> Tuple[list, float]:
    """
    Proposes a new state based on the inital state phi0 (Here 1D floats).
    
    Write helper.

    Args:
        N (int):
            Number of samples that will be drawn
        P (function):
            Distribution function that the samples should follow
        trial_move (function):
            trial move as defined in the Metropolis-Hastings-Algo. 
            Decides wether to use the new state or not.
        initial_config (np.ndarray):
            initial state that acts as a reference value for the trial_state

    Returns:
        states_list, acceptance_rate (Tuple[list, float]):
            list: list of consecutively drawn states
            float: resulting acceptance rate during metropolis-algo  
    """
    acceptance_number = 0

    current_state = initial_config.copy()
    states_list = [current_state]
    for iteration in tqdm(range(N), desc='MC Sampling iteration: '):
        trial_state = trial_move(current_state)
        r = np.random.uniform(0.0, 1.0)

        if r < np.min([1.0, P(trial_state) / P(current_state)]):
            current_state = trial_state.copy()
            acceptance_number += 1.0
        
        states_list.append(current_state) 

    return np.asarray(states_list), acceptance_number/N


if __name__=="__main__":

    l_box = np.array([4, 4])
    N = 10000
    temp_list = np.arange(1.0, 5.1, 0.1).round(2)

    total_state_space = state_space(l_box)
    random_intial_configuration = total_state_space[np.random.randint(0, 2**np.prod(l_box))]

    energy_per_site_list = []
    magnetization_per_site_list = []
    acceptance_rate_list = []
    for temp in temp_list:
        configuration_dynamics, acceptance_rate = metropolis(
            N, 
            functools.partial(boltzmann_probability, temp, l_box),
            ising_trial_move,
            random_intial_configuration,
        )
        energy_per_site, magnetization_per_site = evaluate_observables(configuration_dynamics, l_box)

        energy_per_site_list.append(energy_per_site)
        magnetization_per_site_list.append(magnetization_per_site)
        acceptance_rate_list.append(acceptance_rate)
    
    plt.plot(temp_list, energy_per_site_list, 'x')
    plt.plot(temp_list, magnetization_per_site_list, 'x')
    plt.show()



    