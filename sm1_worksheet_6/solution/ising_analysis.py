"""
TODO:
    - error correction (PLUS correlation time?!)
"""

import cising
import numpy as np
import matplotlib.pyplot as plt

import logging
from tqdm import tqdm

import time
logging.basicConfig(level=logging.INFO)


def calc_observables(beta, L):
    ising = cising.IsingModel(beta, L)

    number_of_samples = 10000
    correlation_time = 100
    equilibration_time = 50000

    # equilibration
    ising.try_many_random_flips(equilibration_time)

    energies_array = np.zeros(number_of_samples)
    magnetization_array = np.zeros(number_of_samples)
    for N in range(number_of_samples):
        ising.try_many_random_flips(correlation_time)

        energies_array[N] = ising.energy()/(L*L)
        magnetization_array[N] = ising.magnetization()

    plt.plot(energies_array)
    plt.plot(magnetization_array)
    plt.show()

    return energies_array.mean(), energies_array.std(), magnetization_array.mean(), magnetization_array.std() 


if __name__=="__main__":
    beta_array = 1/np.arange(1.0, 5.1, 0.1, dtype=float)
    L_array = np.array([16, 64])
    
    L_current = L_array[0]

    energy_mean_list, energy_std_list = [], []
    magnetization_mean_list, magnetization_std_list = [], []
    beta_list = []
    for beta in tqdm(beta_array[::5], desc='beta: '):
        start_time = time.time()
        energy_mean, energy_std, magnetization_mean, magnetization_std = calc_observables(beta, L_current)
        
        beta_list.append(beta)

        energy_mean_list.append(energy_mean)
        energy_std_list.append(energy_std)
        magnetization_mean_list.append(magnetization_mean)
        magnetization_std_list.append(magnetization_std)

        logging.info(f'Duration: {round(time.time() - start_time, 3)} s')

    plt.errorbar(beta_list, energy_mean_list, yerr=energy_std_list, marker='o', label='energies')
    plt.errorbar(beta_list, magnetization_mean_list, yerr=magnetization_std_list, marker='o', label='magnetization')

    plt.legend()
    plt.show()



    