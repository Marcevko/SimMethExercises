import numpy as np
import matplotlib.pyplot as plt

import os
from typing import Tuple, Callable

from functools import partial


def distribution_function(x: float) -> float:
    return np.exp(-x**2)/np.sqrt(np.pi)


def trial_move(delta_x: float, phi0: float) -> float:
    """
    write helper
    """

    r = np.random.uniform(-delta_x, delta_x, size=1)
    return phi0 + r


def metropolis(
        N: int,
        P: Callable, 
        trial_move: Callable, 
        phi0: float,
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
        phi0 (float):
            initial state that acts as a reference value in trial_move

    Returns:
        states_list, acceptance_rate (Tuple[list, float]):
            list: list of consecutively drawn states
            float: resulting acceptance rate during metropolis-algo  
    """
    states_list = np.zeros(N, dtype=float)
    acceptance_list = np.zeros(N, dtype=float)

    current_state = phi0
    for iteration in range(N):
        trial_state = trial_move(phi0)
        r = np.random.uniform(0.0, 1.0, size=1)

        if r < (P(trial_state) / P(current_state)):
            new_state = trial_state

            acceptance_list[iteration] = 1.0
        else: # rejected
            new_state = current_state
        
        states_list[iteration] = new_state
        current_state = new_state

    return states_list, acceptance_list.mean()

if __name__=="__main__":
    # generate 4 sample list with N=100000
    N = 100000
    delta_x_list = [0.1, 1.0, 10.0, 100.0]
    metropolis_samples = dict()

    for delta_x in delta_x_list:
        partial_trial_move = partial(trial_move, delta_x)
        metropolis_sample, acceptance_rate = metropolis(
            N,
            distribution_function,
            partial_trial_move,
            0.5,
        )
        metropolis_samples[delta_x] = {
            f'samples': metropolis_sample,
            f'acceptance_rate': acceptance_rate,
        }

    fig, axs = plt.subplots(2, 2, figsize=(8.0, 6.0))
    axs = axs.flatten()

    color_list = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00']
    
    for indx, (delta_x, sample_dict) in enumerate(metropolis_samples.items()):
        axs[indx].hist(
            sample_dict['samples'], 
            bins=100,
            range=(-5.0, 5.0), 
            label=r'$\Delta x = $' + f'{delta_x}' +f'; acc_rate = {sample_dict["acceptance_rate"]}',
            color=color_list[indx],
        )

        axs[indx].legend(loc='upper right')
    plt.show()

