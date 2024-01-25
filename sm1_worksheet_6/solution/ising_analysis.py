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


def calc_observables(beta, L, number_of_samples=None, corr_time=None, equib_time=None, plot_dir='observable_dynamics'):
    ising = cising.IsingModel(beta, L)

    number_of_samples = 10000 if number_of_samples is None else number_of_samples
    correlation_time = 100 if corr_time is None else corr_time
    equilibration_time = 1e4 if equib_time is None else equib_time

    # equilibration
    for n in range(int(equilibration_time)):
        ising.try_many_random_flips(corr_time)

    energies_array = np.zeros(number_of_samples)
    magnetization_array = np.zeros(number_of_samples)
    for N in range(number_of_samples):
        ising.try_many_random_flips(correlation_time)

        energies_array[N] = ising.energy()/(L*L)
        magnetization_array[N] = ising.magnetization()

    plt.plot(energies_array, label='energy per lattice site')
    plt.plot(magnetization_array, label='magnetization per site')
    plt.xlabel('integration steps')
    plt.ylabel('Observable magnitude [a.u.]')
    plt.title(f'Observable dynamics for T={round(1/beta, 2)}, L={L}')
    plt.legend()
    plt.savefig(f'./sm1_worksheet_6/plots/{plot_dir}/T_{round(1/beta, 2)}_L_{L}.png', dpi=150)
    plt.close()

    return energies_array, magnetization_array


def generate_data(temp_array, L, number_of_samples=None, corr_time=None, equip_time=None):
    energy_mean_list, energy_std_list = [], []
    magnetization_mean_list, magnetization_std_list = [], []
    
    for beta in tqdm(1/temp_array, desc='Iterating over all temperatures: '):
        start_time = time.time()
        energies_arr, magnetizations_arr = calc_observables(
            beta,
            L, 
            number_of_samples=number_of_samples, 
            corr_time=corr_time, 
            equib_time=equip_time,
        )
        energy_mean, energy_std = energies_arr.mean(), energies_arr.std()
        magnetization_mean, magnetization_std = magnetizations_arr.mean(), magnetizations_arr.std()

        energy_mean_list.append(energy_mean)
        energy_std_list.append(energy_std)                 
        magnetization_mean_list.append(magnetization_mean)       
        magnetization_std_list.append(magnetization_std)                                                              
                            
        logging.info(f'Duration: {round(time.time() - start_time, 3)} s')

        np.save(
            f'./sm1_worksheet_6/data/observable_dynamics/T_{round(1/beta, 2)}_L_{L}.npy', np.array([energies_arr, magnetizations_arr])
        )

    np.save(
        f'./sm1_worksheet_6/data/T_dependent_observables_L_{L}.npy', np.array([energy_mean_list, magnetization_mean_list])
    )


if __name__=="__main__":
    temp_array_ex1 = np.arange(1.0, 5.1, 0.1, dtype=float)
    L_array_ex1 = np.array([16, 64])
    corr_times_ex1 = [50, 100]
    equib_times_ex1 = [5e4, 1e5]
    
    temp_array_ex2 = np.arange(2.0, 2.41, 0.02, dtype=float)
    L_array_ex2 = np.array([4, 16, 32])
    num_samples_ex2 = np.array([10000, 20000, 50000])
    corr_times_ex2 = [100, 100, 100]
    equib_times_ex2 = [1e6, 1e7, 1e9]

    GENERATE_EX_1 = True
    GENERATE_EX_2 = False    

    # generating data for ex1
    if GENERATE_EX_1:
        for indx, L in enumerate(L_array_ex1):
            generate_data(
                temp_array_ex1, 
                L,
                corr_time=corr_times_ex1[indx],
                equip_time=equib_times_ex1[indx],
            )

    # plot T-dependent observables
    
    # generate data for ex2
    if GENERATE_EX_2:
        for indx, L in enumerate(L_array_ex2):
            generate_data(
                temp_array_ex2,
                L,
                number_of_samples=num_samples_ex2[indx],
                corr_time=corr_times_ex2[indx],
                equip_time=equib_times_ex2[indx],
            )






    