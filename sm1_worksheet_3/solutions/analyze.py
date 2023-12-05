import pickle
import argparse

import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple

parser = argparse.ArgumentParser()
parser.add_argument('file', help="Path to pickle file.")
args = parser.parse_args()

with open(args.file, 'rb') as fp:
    data = pickle.load(fp)

def running_average(O: np.ndarray, M: int) -> np.ndarray:
    """
    Computes the running average of the np.array O with M being the half-window-size.
    
    Currently only works for 1d arrays. Could be become an issue.
    """
    array_length = len(O)
    output = np.full(array_length, np.nan)
    for i in range(M, array_length - M):
        output[i] = np.sum(O[i - M: i + M + 1], axis=0) / (2*M + 1)
    
    return output

def compute_equilibrated_observable_mean(O: np.ndarray, t_eq: int):
    """
    Computes the mean of the observable O after the equiblibration time t_eq (the integration step).
    """
    mean_value = np.nanmean(O[t_eq:])
    return mean_value

def compute_rdf(rdfs: list) -> np.ndarray:
    rdfs = np.asarray(rdfs)

    rdf = np.mean(rdfs, axis=0)
    return rdf

# create plots for exercise 5
if args.file == './sm1_worksheet_3/checkpoints/ex_5_checkpoint.pkl':
    fig, axs = plt.subplots(3, 1, figsize=(6.0, 17.0))
    mathematical_symbols = ['E', 'T', 'P']

    for indx, observable in enumerate(['energies', 'temperatures', 'pressures']):
        unaveraged_data  = np.array(data[observable])
        averaged_data_10 = running_average(unaveraged_data, 10)
        averaged_data_100 = running_average(unaveraged_data, 100)

        time_array = 0.03 * np.arange(0.0, len(unaveraged_data), 1)

        axs[indx].plot(time_array, unaveraged_data, label=f'unaveraged {observable}')
        axs[indx].plot(time_array, averaged_data_10, label=f'running average with M=10')
        axs[indx].plot(time_array, averaged_data_100, label=f'running average with M=100')
        
        legend_loc = 'upper right' if indx != 1 else 'lower right'
        axs[indx].legend(loc=legend_loc)
        axs[indx].set_xlabel(r'time $t$')
        axs[indx].set_ylabel(f'{observable} {mathematical_symbols[indx]}')

        equilibrated_mean_observable = compute_equilibrated_observable_mean(unaveraged_data, int(500*100/3))
        print(f'Mean of {observable} after equilibration time t_eq=500: {equilibrated_mean_observable}')

    fig.tight_layout(w_pad=2.0)
    plt.savefig('sm1_worksheet_3/plots/running_averages.png', format='png', dpi=600)
    plt.show()

# create plots for exercise 6
if './sm1_worksheet_3/checkpoints/ex_6_checkpoint' in args.file:
    fig, axs = plt.subplots(3, 1, figsize=(6.0, 17.0))
    mathematical_symbols = ['E', 'T', 'P']

    for indx, observable in enumerate(['energies', 'temperatures', 'pressures']):
        unaveraged_data  = np.array(data[observable])
        averaged_data_10 = running_average(unaveraged_data, 10)
        averaged_data_100 = running_average(unaveraged_data, 100)

        time_array = 0.03 * np.arange(0.0, len(unaveraged_data), 1)

        axs[indx].plot(time_array, unaveraged_data, label=f'unaveraged {observable}')
        axs[indx].plot(time_array, averaged_data_10, label=f'running average with M=10')
        axs[indx].plot(time_array, averaged_data_100, label=f'running average with M=100')
        
        legend_loc = 'upper right' if indx != 1 else 'lower right'
        axs[indx].legend(loc=legend_loc)
        axs[indx].set_xlabel(r'time $t$')
        axs[indx].set_ylabel(f'{observable} {mathematical_symbols[indx]}')

    fig.tight_layout(w_pad=2.0)
    splitted_string = args.file.split('.')
    plt.savefig(f'sm1_worksheet_3/plots/running_averages_{splitted_string[1][-2:]}.png', format='png', dpi=600)
    plt.show()

# create test plots for exercise 7
if args.file=='./sm1_worksheet_3/checkpoints/ex_7_checkpoint.pkl':
    forces = np.asarray(data['forces_all'])
    
    time_array = 0.03 * np.arange(0.0, len(forces), 1)

    plt.plot(time_array, forces[:, 0, :])
    
    plt.title('time evolution of forces (x-coordinate)')
    plt.xlabel(r'time $t$')
    plt.ylabel(r'forces $F_x$')
    plt.savefig('./sm1_worksheet_3/plots/force_capping_plot.png', format='png', dpi=600)
    plt.show()

# create plots for exercise 8
if './sm1_worksheet_3/checkpoints/ex_8_checkpoint' in args.file:
    rdfs_loaded = data['rdfs']
    test = compute_rdf(rdfs_loaded)

    x_array = np.linspace(0.8, 5.0, 100)
    plt.plot(x_array, test)
    plt.show()

# create plots for exercise 9
# if './sm1_worksheet_3/checkpoints/ex_9_checkpoint' in args.file:
    # forces = np.asarray(data['forces_all'])
    
    # time_array = 0.03 * np.arange(0.0, len(forces), 1)

    # plt.plot(time_array, forces[:, 0, :])
    # plt.xlim([0, 5])
    # plt.ylim([-500, 500])
    
    # plt.title('time evolution of forces (x-coordinate)')
    # plt.xlabel(r'time $t$')
    # plt.ylabel(r'forces $F_x$')
    # plt.show()

# create plots for exercise 9
if './sm1_worksheet_3/checkpoints/ex_9_checkpoint' in args.file:
    fig, axs = plt.subplots(3, 1, figsize=(6.0, 17.0))
    mathematical_symbols = ['E', 'T', 'P']

    for indx, observable in enumerate(['energies', 'temperatures', 'pressures']):
        unaveraged_data  = np.array(data[observable])
        averaged_data_10 = running_average(unaveraged_data, 10)
        averaged_data_100 = running_average(unaveraged_data, 100)

        time_array = 0.03 * np.arange(0.0, len(unaveraged_data), 1)

        axs[indx].plot(time_array, unaveraged_data, label=f'unaveraged {observable}')
        axs[indx].plot(time_array, averaged_data_10, label=f'running average with M=10')
        axs[indx].plot(time_array, averaged_data_100, label=f'running average with M=100')
        
        legend_loc = 'upper right' if indx != 1 else 'lower right'
        axs[indx].legend(loc=legend_loc)
        axs[indx].set_xlabel(r'time $t$')
        axs[indx].set_ylabel(f'{observable} {mathematical_symbols[indx]}')

    fig.tight_layout(w_pad=2.0)
    splitted_string = args.file.split('.')
    plt.savefig(f'sm1_worksheet_3/plots/equilibration_{splitted_string[1][-2:]}.png', format='png', dpi=600)
    plt.show()


    rdfs_loaded = data['rdfs']
    meaned_rdf = compute_rdf(rdfs_loaded)
    rdf_time_array = np.linspace(0.8, 5.0, 100)
    plt.plot(rdf_time_array, meaned_rdf)
    plt.title(f'RDF for T={splitted_string[1][-2]}.{splitted_string[1][-1]}')
    plt.xlabel(r'distance $r$')
    plt.ylabel(r'$g(r)$')

    plt.savefig(f'sm1_worksheet_3/plots/RDF_{splitted_string[1][-2:]}.png', format='png', dpi=600)
    plt.show()


if args.file=='./sm1_worksheet_3/checkpoints/ex_7_checkpoint_test.pkl':
    forces = np.asarray(data['forces_all'])
    
    time_array = 0.03 * np.arange(0.0, len(forces), 1)

    plt.plot(time_array, forces[:, 0, :])
    # plt.xlim([0, 5])
    # plt.ylim([-500, 500])
    
    plt.title('time evolution of forces (x-coordinate)')
    plt.xlabel(r'time $t$')
    plt.ylabel(r'forces $F_x$')
    # plt.savefig('./sm1_worksheet_3/plots/force_capping_plot.png', format='png', dpi=600)
    plt.show()