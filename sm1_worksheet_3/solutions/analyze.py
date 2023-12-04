import pickle
import argparse

import matplotlib.pyplot as plt
import numpy as np

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
