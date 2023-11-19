#!/usr/bin/env python3

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.optimize import curve_fit

import ex_3_4


def init_2dgrid_positions(n_per_side: float, box: np.ndarray) -> np.ndarray:
    
    n_part = n_per_side**2
    x = np.zeros((2, n_part))

    xgrid_positions = np.linspace(0.0, box[0], n_per_side + 1)[:-1] + 0.5*(box[0] // n_per_side)
    ygrid_positions = np.linspace(0.0, box[1], n_per_side + 1)[:-1] + 0.5*(box[1] // n_per_side)

    x[0, :] = np.tile(xgrid_positions, n_per_side)
    x[1, :] = np.repeat(ygrid_positions, n_per_side)
    
    return x


def init_and_run_sim(n_per_side: float) -> Tuple[int, float]:
    DT = 0.01
    T_MAX = 1.0
    N_TIME_STEPS = int(T_MAX / DT)

    R_CUT = 2.5
    SHIFT = 0.016316891136

    DENSITY = 0.7
    N_PART = n_per_side**2
    VOLUME = N_PART / DENSITY
    BOX = np.ones(2) * VOLUME**(1. / 2.)

    # particle positions

    # SET UP THE PARTICLE POSITIONS ON A LATTICE HERE
    x = init_2dgrid_positions(n_per_side, BOX)

    # if n_per_side == 5:
    #     plt.plot(x[0, :], x[1, :], '.', color='red')
    #     plt.axhline(y=0.0, ls=':', color='k')
    #     plt.axhline(y=BOX[1], ls=':', color='k')
    #     plt.axvline(x=0.0, ls=':', color='k')
    #     plt.axvline(x=BOX[0], ls=':', color='k')
    #     plt.savefig('sm1_worksheet_2/plots/grid_positions_5_particles.png', format='png', dpi=600)
        # plt.show()

    # random particle velocities
    v = 2.0 * np.random.random((2, N_PART)) - 1.0

    f = ex_3_4.forces(x, R_CUT, BOX)

    positions = np.full((N_TIME_STEPS, 2, N_PART), np.nan)
    energies = np.full((N_TIME_STEPS), np.nan)

    start_time = time.time()

    for i in range(N_TIME_STEPS):
        x, v, f = ex_3_4.step_vv(x, v, f, DT, R_CUT, BOX)

        positions[i] = x
        energies[i] = ex_3_4.total_energy(x, v, R_CUT, BOX)

    end_time = time.time()
    run_time = end_time - start_time
    print(f'{N_PART}, {run_time}')

    return N_PART, run_time


def parabola(x, a):
    return a*x**2


if __name__ == "__main__":

    n_per_side_array = np.arange(3, 13, 1, dtype=int)

    particle_numbers, simulation_runtimes = [], []
    for n in n_per_side_array:
        part_number, sim_runtime = init_and_run_sim(n)
        particle_numbers.append(part_number)
        simulation_runtimes.append(sim_runtime)
    
    # fit
    popt, pcov = curve_fit(parabola, particle_numbers, simulation_runtimes)
    print(f'Fit finished!')
    print(f'popt, pcov: {popt}, {pcov}')

    fig, axs = plt.subplots(2, 2, figsize=(12.0, 9.0))
    
    for index1 in range(2):
        for index2 in range(2):
            axs[index1, index2].plot(particle_numbers, simulation_runtimes, lw=1, ls=":", color='mediumblue', marker='o', markersize=6)
    

    axs[0, 0].set_title('vanilla', fontsize=11)

    axs[1, 1].set_title('doublelog', fontsize=11)
    axs[1, 1].loglog()
    
    axs[0, 1].set_title('semilog x', fontsize=11)
    axs[0, 1].semilogx()

    axs[1, 0].set_title('semilog y', fontsize=11)
    axs[1, 0].semilogy()

    axs[1, 0].set_xlabel('Number of particles')
    axs[1, 1].set_xlabel('Number of particles')

    axs[0, 0].set_ylabel('Simulation runtime [s]')
    axs[1, 0].set_ylabel('Simulation runtime [s]')

    fig.suptitle(f'Simulation Runtime vs. Particle Number')
    fig.tight_layout(w_pad=2.0)
    plt.savefig('sm1_worksheet_2/plots/runtime_multiplot_nofit.png', format='png', dpi=600)
    plt.show()


    x_fit_array = np.linspace(6.0, 180.0, 5000)
    fig, axs = plt.subplots(2, 2, figsize=(12.0, 9.0))

    for index1 in range(2):
        for index2 in range(2):
            axs[index1, index2].plot(x_fit_array, parabola(x_fit_array, *popt), lw=1, color='lightskyblue', alpha=1.0, label=f'parabolic fit, a={popt}')
            axs[index1, index2].plot(particle_numbers, simulation_runtimes, 'o', color='mediumblue', markersize=6, label='measured runtimes')
    

    axs[0, 0].set_title('vanilla', fontsize=11)
    axs[0, 0].legend(loc='upper left')

    axs[1, 1].set_title('doublelog', fontsize=11)
    axs[1, 1].loglog()
    
    axs[0, 1].set_title('semilog x', fontsize=11)
    axs[0, 1].semilogx()

    axs[1, 0].set_title('semilog y', fontsize=11)
    axs[1, 0].semilogy()

    axs[1, 0].set_xlabel('Number of particles')
    axs[1, 1].set_xlabel('Number of particles')

    axs[0, 0].set_ylabel('Simulation runtime [s]')
    axs[1, 0].set_ylabel('Simulation runtime [s]')

    fig.suptitle(f'Simulation Runtime vs. Particle Number')
    fig.tight_layout(w_pad=2.0)
    plt.savefig('sm1_worksheet_2/plots/runtime_multiplot_fitted.png', format='png', dpi=600)
    plt.show()