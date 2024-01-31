"""
TODO:
    - error correction (PLUS correlation time?!)
"""

import cising
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import logging
from tqdm import tqdm

import pickle
import gzip

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

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

def compute_binder_parameter(magnetization_array):
    
    term = np.power(magnetization_array, 4).mean() / (np.power(magnetization_array, 2).mean())**2
    
    return 1 - (1/3) * term



def generate_data(temp_array, L, number_of_samples=None, corr_time=None, equip_time=None, plot_dir='observable_dynamics'):
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
            plot_dir=plot_dir,
        )
        energy_mean, energy_std = energies_arr.mean(), energies_arr.std()
        magnetization_mean, magnetization_std = magnetizations_arr.mean(), magnetizations_arr.std()

        energy_mean_list.append(energy_mean)
        energy_std_list.append(energy_std)                 
        magnetization_mean_list.append(magnetization_mean)       
        magnetization_std_list.append(magnetization_std)                                                              
                            
        logging.info(f'Duration: {round(time.time() - start_time, 3)} s')

        # np.save(
        #     f'./sm1_worksheet_6/data/{plot_dir}/T_{round(1/beta, 2)}_L_{L}.npy', np.array([energies_arr, magnetizations_arr])
        # ) 
        savefile = np.array([energies_arr, magnetizations_arr])
        filepath = f'./sm1_worksheet_6/data/{plot_dir}/T_{round(1/beta, 2)}_L_{L}.pkl.gz'
        with gzip.open(filepath, 'wb') as file:
            pickle.dump(savefile, file)
        
        logging.info(f'Data saved successfully!')


    np.save(
        f'./sm1_worksheet_6/data/T_{plot_dir}_L_{L}.npy', 
        np.array([energy_mean_list, energy_std_list, magnetization_mean_list, magnetization_std_list])
    )


def find_intersection(func1, func2, initial_guess):
    def func_difference(x, func1, func2):
        return func1(x) - func2(x)

    intersection = fsolve(func_difference, initial_guess, args=(func1, func2))
    return intersection



if __name__=="__main__":
    temp_array_ex1 = np.arange(1.0, 5.1, 0.1, dtype=float)
    L_array_ex1 = np.array([16, 64])
    num_samples_ex1 = np.array([50000, 50000])
    corr_times_ex1 = [50, 100]
    equib_times_ex1 = [5e4, 5e6]
    
    temp_array_ex2 = np.arange(2.0, 2.41, 0.02, dtype=float)
    L_array_ex2 = np.array([4, 16, 32])
    num_samples_ex2 = np.array([50000, 5e6, 1e7], dtype=int)
    corr_times_ex2 = [100, 50, 50]
    equib_times_ex2 = [1e5, 5e6, 1e7]

    GENERATE_EX_1 = False
    GENERATE_EX_2 = False
    GENERATE_EX_3 = True

    # generating data for ex1
    if GENERATE_EX_1:
        for indx, L in enumerate(L_array_ex1):
            generate_data(
                temp_array_ex1, 
                L,
                number_of_samples=num_samples_ex1[indx],
                corr_time=corr_times_ex1[indx],
                equip_time=equib_times_ex1[indx],
                plot_dir='observable_dynamics'
            )
    
    # generate data for ex2
    if GENERATE_EX_2:
        for indx, L in enumerate(L_array_ex2):
            generate_data(
                temp_array_ex2,
                L,
                number_of_samples=num_samples_ex2[indx],
                corr_time=corr_times_ex2[indx],
                equip_time=equib_times_ex2[indx],
                plot_dir='binder_observables'
            )

    # generating ex_1 plot
    import matplotlib.patches as mpatches

    exact_results = np.load('./sm1_worksheet_6/data/ising_exact.npy')
    mc_l_16 = np.load('./sm1_worksheet_6/data/T_observable_dynamics_L_16.npy')
    mc_l_64 = np.load('./sm1_worksheet_6/data/T_observable_dynamics_L_64.npy')

    fig = plt.figure(figsize=(8, 6))
    plt.plot(
        temp_array_ex1,
        exact_results[0],
        label='L=4 (exact energies)',
        color='tomato'
    )
    plt.plot(
        temp_array_ex1,
        exact_results[1],
        color='dodgerblue',
        label='L=4 (exact magnetization)'
        )
    
    plt.errorbar(
        temp_array_ex1, 
        mc_l_16[0],
        yerr=mc_l_16[1],
        linestyle='',
        marker='s',
        label='L=16 (MC energies)',
        color='red',
        lolims=True,
        uplims=True,
    )
    plt.errorbar(
        temp_array_ex1, 
        mc_l_16[2],
        yerr=mc_l_16[3],
        linestyle='',
        marker='s',
        label='L=16 (MC magnetization)',
        color='darkturquoise',
        lolims=True,
        uplims=True,
    )

    plt.errorbar(
        temp_array_ex1, 
        mc_l_64[0],
        yerr=mc_l_64[1],
        linestyle='',
        marker='o',
        label='L=64 (MC energies)',
        color='darkred',
        lolims=True,
        uplims=True,
    )
    plt.errorbar(
        temp_array_ex1, 
        mc_l_64[2],
        yerr=mc_l_64[3],
        linestyle='',
        marker='o',
        label='L=64 (MC magnetization)',
        color='navy',
        lolims=True,
        uplims=True,
    )

    handels, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handels, 
        labels, 
        loc='lower center',
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
    )
    plt.title('Equilibrated Observables for different Ising model sizes L')
    plt.xlabel(r'Temperature $T$')
    plt.ylabel(r'Observable $m$ or $e$ [a.u.]')
    fig.tight_layout()
    plt.savefig('./sm1_worksheet_6/plots/ex_1_L_comparison.png', dpi=150)
    plt.show()

    # plotting the binder parameter
    # for L in L_array_ex2:
    #     binder_list =  []
    #     for temp in temp_array_ex2:
    #         filepath = f'./sm1_worksheet_6/data/binder_observables/T_{round(temp, 2)}_L_{L}.pkl.gz'
    #         with gzip.open(filepath, 'rb', ) as f:
    #             dataset = pickle.load(f)
    #             binder_param = compute_binder_parameter(dataset[1])
    #             binder_list.append(binder_param)
    #     np.save(f'./sm1_worksheet_6/data/binder_L_{L}.npy', np.array(binder_list))

    binder_l_4_list = np.load(f'./sm1_worksheet_6/data/binder_L_4.npy')
    binder_l_16_list = np.load(f'./sm1_worksheet_6/data/binder_L_16.npy')
    binder_l_32_list = np.load(f'./sm1_worksheet_6/data/binder_L_32.npy')

    # interpolate the data
    def interpolate_binder_params(x_arr: np.ndarray, y_arr: np.ndarray):
        interpolated_function = interp1d(x_arr, y_arr, kind='cubic')
        return interpolated_function
    
    interpolation_x_locations = np.linspace(2.0, 2.40, int(1e5))
    binder_l_4_interpolated = interpolate_binder_params(temp_array_ex2, binder_l_4_list)(interpolation_x_locations)
    binder_l_16_interpolated = interpolate_binder_params(temp_array_ex2, binder_l_16_list)(interpolation_x_locations)
    binder_l_32_interpolated = interpolate_binder_params(temp_array_ex2, binder_l_32_list)(interpolation_x_locations)
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(temp_array_ex2, binder_l_4_list, 'o', label='L=4', color='mediumblue')
    plt.plot(interpolation_x_locations, binder_l_4_interpolated, label='L=4, interpolated', color='mediumblue')
    
    plt.plot(temp_array_ex2, binder_l_16_list, 'o', label='L=16', color='darkred')
    plt.plot(interpolation_x_locations, binder_l_16_interpolated, label='L=16, interpolated', color='darkred')

    plt.plot(temp_array_ex2, binder_l_32_list, 'o', label='L=32', color='darkturquoise')
    plt.plot(interpolation_x_locations, binder_l_32_interpolated, label='L=32, interpolated', color='darkturquoise')

    plt.title(r'Binder Parameter $U$ vs. System Temperature $T$')
    plt.xlabel(r'Temperature $T$')
    plt.ylabel(r'Binder Parameter $U$')

    handels, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
            handels, 
            labels, 
            loc='lower center',
            bbox_to_anchor=(0.5, -0.25),
            ncol=3,
        )

    axin = inset_axes(plt.gca(), width="100%", height="100%", loc='center', bbox_to_anchor=(0.15, 0.18, 0.45, 0.5), bbox_transform=plt.gca().transAxes)

    axin.plot(temp_array_ex2, binder_l_4_list, 'o', color='mediumblue')
    axin.plot(interpolation_x_locations, binder_l_4_interpolated, color='mediumblue')

    axin.plot(temp_array_ex2, binder_l_16_list, 'o', color='darkred')
    axin.plot(interpolation_x_locations, binder_l_16_interpolated, color='darkred')

    axin.plot(temp_array_ex2, binder_l_32_list, 'o', color='darkturquoise')
    axin.plot(interpolation_x_locations, binder_l_32_interpolated, color='darkturquoise')

    axin.legend().set_visible(False)
    axin.set_xlim([2.25, 2.27])
    axin.set_ylim([0.61, 0.625])
    axin.set_xlabel('')  
    axin.set_ylabel('')
    axin.set_title('')

    
    fig.tight_layout()
    plt.savefig('./sm1_worksheet_6/plots/binder_parameter.png', dpi=150)
    plt.show()

    # calculate intersections
    initial_guess = 2.26
    intersec_l_4_16 = find_intersection(
        interpolate_binder_params(temp_array_ex2, binder_l_4_list), 
        interpolate_binder_params(temp_array_ex2, binder_l_16_list),
        initial_guess,    
    )   
    intersec_l_4_32 = find_intersection(
        interpolate_binder_params(temp_array_ex2, binder_l_4_list), 
        interpolate_binder_params(temp_array_ex2, binder_l_32_list),
        initial_guess,    
    )
    intersec_l_16_32 = find_intersection(
        interpolate_binder_params(temp_array_ex2, binder_l_16_list), 
        interpolate_binder_params(temp_array_ex2, binder_l_32_list),
        initial_guess,    
    )
    binder_intersections = np.array([intersec_l_4_16, intersec_l_4_32, intersec_l_16_32])
    np.save('./sm1_worksheet_6/data/binder_intersections.npy', binder_intersections)

    critical_temp_calc, critical_temp_error = binder_intersections.mean(), binder_intersections.std()
    print(f'estimated Critical Temperature: {critical_temp_calc}')
    print(f'estimated error: {critical_temp_error}')

    critical_temp_theo = 2/(np.log(1 + np.sqrt(2)))
    print(f'Theoretical Critical Temperature: {critical_temp_theo}')

    # generating data for 4.2
    temp_array_ex_3_calc = np.array([critical_temp_calc])
    temp_array_ex_3_theo = np.array([critical_temp_theo])
    L_array_ex3 = np.array([8, 16, 32, 64, 128], dtype=int)
    num_samples_ex3 = np.array([1e5, 5e5, 1e6, 5e6, 1e7], dtype=int)
    corr_times_ex3 = np.array([100, 100, 50, 50, 50], dtype=int)
    equib_times_ex3 = np.array([5e4, 5e4, 5e5, 5e6, 2e7], dtype=int)

    if GENERATE_EX_3:
        for indx, L in enumerate(L_array_ex3):
            print(L)
            generate_data(
                temp_array_ex_3_calc,
                L,
                number_of_samples=num_samples_ex3[indx],
                corr_time=corr_times_ex3[indx],
                equip_time=equib_times_ex3[indx],
                plot_dir='beta_estimation_calc_T',
            )
            generate_data(
                temp_array_ex_3_theo,
                L,
                number_of_samples=num_samples_ex3[indx],
                corr_time=corr_times_ex3[indx],
                equip_time=equib_times_ex3[indx],
                plot_dir='beta_estimation_theo_T',
            )
