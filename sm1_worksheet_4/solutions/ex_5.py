"""
TODO:
- extend code so that all particles are considered (only left for the VACF!)

- fitting for MSD
- numeric integration for VACF

- what is meant with the scaling in VACF
"""

import gzip
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# all written for only one atom, add atom_indx as argument
def compute_autocorrelation(vels: np.ndarray, lag_time: int, atom_indx: int):
    """
    write helper
    """
    vel_indx = 0
    correlation_list = []
    # running true every point and computing the dot-product of starting_velocity and lag_time_velocity
    while vel_indx < (vels.shape[0] - lag_time):
        correlation_list.append(
            np.dot(vels[vel_indx, atom_indx, :], vels[vel_indx + lag_time, atom_indx, :])
        )
        vel_indx += 1
    return np.mean(correlation_list)

def compute_VACF(vels: np.ndarray, atom_indx: int) -> np.ndarray:
    """
    Computes the VACF (np.ndarray of shape shape as input) using the implementation of the autocorrelation function.
    """
    max_lag_time = vels.shape[0] // 2
    autocorrelation_list = []
    lag_time_list = np.arange(1, max_lag_time + 1, dtype=int)
    for lag_time in lag_time_list:
        autocorrelation_list.append(compute_autocorrelation(vels, lag_time, atom_indx))
    
    return np.array(autocorrelation_list), lag_time_list

def compute_MSD(traj: np.ndarray, lag_time: int, atom_indx: int) -> np.ndarray:
    """
    Computes the ensebmle averaged MSD for one lag_time
    """
    traj_indx = 0
    msd_list = []
    while traj_indx < (traj.shape[0] - lag_time):
        distance_vector = traj[traj_indx + lag_time, atom_indx, :] - traj[traj_indx, atom_indx, :]
        msd_list.append(np.dot(distance_vector, distance_vector))
        traj_indx += 1

    return np.mean(msd_list), np.std(msd_list)

def compute_einstein_diffusion(traj: np.ndarray, atom_indx: int):
    """
    write helper
    """
    max_lag_time = traj.shape[0] // 2
    einstein_relation_list, einstein_relation_errorbars = [], []

    lag_time_list = np.arange(1, max_lag_time + 1, dtype=int)
    for lag_time in lag_time_list:
        msd, errorbar = compute_MSD(traj, lag_time, atom_indx)
        einstein_relation_list.append(msd)
        einstein_relation_errorbars.append(errorbar)

    return np.array(einstein_relation_list), lag_time_list, np.array(einstein_relation_errorbars)

if __name__ == '__main__':
    # loading data
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help="Path to pickle file.")
    args = parser.parse_args()
    
    with gzip.open(args.file, 'rb') as datafile:
        data = pickle.load(datafile)
        
        N, T, GAMMA_LANGEVIN, x, v, ts, Es, Tms, vels, traj = data
    
    # compute MSD
    plt.plot(traj[:, 0, 0])
    plt.show()

    if os.path.isfile('./sm1_worksheet_4/plots/msd_all_particles.npy'):
        msd_list = np.load('./sm1_worksheet_4/plots/msd_all_particles.npy')
        _, lag_time_list, single_atom_errorbars = compute_einstein_diffusion(traj, 0)
    else:
        msd_list = []
        for atom_indx in tqdm(range(traj.shape[1]), 'MSD calc for atom'):
            atom_msd, lag_time_list, single_atom_errorbars = compute_einstein_diffusion(traj, atom_indx)
            msd_list.append(atom_msd)
        msd_list = np.array(msd_list)
        np.save('./sm1_worksheet_4/plots/msd_all_particles.npy', msd_list)

    # compute plot for one atom
    fig, axs = plt.subplots(1, 2, figsize=(14.0, 6.0))
    axs[0].errorbar(lag_time_list, msd_list[0, :], yerr=single_atom_errorbars, ecolor=(0.5, 0.5, 0.5, 0.3))
    axs[0].set_ylabel(r'lag time $\tau$')
    axs[0].set_xlabel(r'MSD($\tau$)')
    axs[0].set_title(r'MSD($\tau$) for the first particle')

    axs[1].errorbar(lag_time_list[:100], msd_list[0, :100], yerr=single_atom_errorbars[:100], ecolor=(0.5, 0.5, 0.5, 0.3))
    axs[1].set_ylabel(r'lag time $\tau$')
    axs[1].set_xlabel(r'$MSD(\tau)$')
    axs[1].set_title(r'MSD($\tau$) for the first particle (zoomed in)')

    fig.tight_layout()
    plt.savefig('./sm1_worksheet_4/plots/MSD_first_particle.png', format='png', dpi=150)
    plt.show()

    # compute plot for all atoms
    msd_all_atoms = np.mean(msd_list, axis=0)
    averaged_erros = single_atom_errorbars / np.sqrt(msd_list.shape[0])

    fig, axs = plt.subplots(1, 2, figsize=(14.0, 6.0))
    axs[0].errorbar(lag_time_list, msd_all_atoms, yerr=averaged_erros, ecolor=(0.5, 0.5, 0.5, 0.3))
    axs[0].set_ylabel(r'lag time $\tau$')
    axs[0].set_xlabel(r'MSD($\tau$)')
    axs[0].set_title(r'MSD($\tau$) for all particles')

    axs[1].errorbar(lag_time_list[:100], msd_all_atoms[:100], yerr=averaged_erros[:100], ecolor=(0.5, 0.5, 0.5, 0.3))
    axs[1].set_ylabel(r'lag time $\tau$')
    axs[1].set_xlabel(r'MSD($\tau$)')
    axs[1].set_title(r'MSD($\tau$) for all particles (zoomed in)')

    fig.tight_layout()
    plt.savefig('./sm1_worksheet_4/plots/MSD_all_particles.png', format='png', dpi=150)
    plt.show()

    # compute VACF
    plt.plot(vels[:, 0, 0])
    plt.show()

    if os.path.isfile('./sm1_worksheet_4/plots/vacf_all_particles.npy'):
        vacf_list = np.load('./sm1_worksheet_4/plots/vacf_all_particles.npy')
        _, lag_time_list = compute_VACF(vels, 0)
    else:
        vacf_list = []
        for atom_indx in tqdm(range(vels.shape[1]), 'VACF calc for Atom'):
            atom_vacf, lag_time_list = compute_VACF(vels, atom_indx)
            vacf_list.append(atom_vacf)
        vacf_list = np.array(vacf_list)
        np.save('./sm1_worksheet_4/plots/vacf_all_particles.npy', vacf_list)

    # compute plot for all atoms
    vacf_all_atoms = np.mean(vacf_list, axis=0)
    averaged_erros = single_atom_errorbars / np.sqrt(vacf_list.shape[1])

    fig, axs = plt.subplots(1, 2, figsize=(14.0, 6.0))
    axs[0].plot(lag_time_list, vacf_all_atoms)
    axs[0].set_ylabel(r'lag time $\tau$')
    axs[0].set_xlabel(r'VACF($\tau$)')
    axs[0].set_title(r'VACF($\tau$) for all particles')

    axs[1].errorbar(lag_time_list[:100], vacf_all_atoms[:100])
    axs[1].set_ylabel(r'lag time $\tau$')
    axs[1].set_xlabel(r'VACF($\tau$)')
    axs[1].set_title(r'VACF($\tau$) for all particles (zoomed in)')

    fig.tight_layout()
    plt.savefig('./sm1_worksheet_4/plots/VACF_all_particles.png', format='png', dpi=150)
    plt.show()
