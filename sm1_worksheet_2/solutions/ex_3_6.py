"""
Notes on the exercise:

    - The copy of the particles is returned, so that the maximal distance between the new (integrated) positions
        and the previous positions can be estimated (to be able to decide whether Verlet update is necessary).

TODO:
    - in der 3.4 die funktion in minimal_image_vector umbennenen
    - main umschreiben, sodass ich ueber skin values iterieren kann und die runtime plots erstellen kann
    - Zaehlen wie haeufig die liste geupdated wurde (in die plots einarbeiten)        
    - PBC are missing in VV I think
    - Iwas stimmt mit der Intergration oder so nicht. Muss mir das ganze noch ein mal ansehen.
"""
import numpy as np
import scipy.linalg
import time

import itertools
import tqdm
import matplotlib.pyplot as plt

from typing import Tuple

import ex_3_4
import ex_3_5

global update_counter
update_counter: int = 0


def forces(x: np.ndarray, r_cut: float, box: np.ndarray, verlet_list: np.ndarray) -> np.ndarray:
    """Compute and return the forces acting onto the particles,
    depending on the positions x."""
    N = x.shape[1]
    f = np.zeros_like(x)
    for pair in verlet_list:
            # distance vector
            r_ij = ex_3_4.minimum_image_vector(x[:, pair[0]], x[:, pair[1]], box) 
            f_ij = ex_3_4.lj_force(r_ij, r_cut)
            f[:, pair[0]] += f_ij
            f[:, pair[1]] -= f_ij
    return f


def total_energy(x: np.ndarray, v: np.ndarray, r_cut: float, shift:float, box: np.ndarray, verlet_list: np.ndarray) -> float:
    """Compute and return the total energy of the system with the
    particles at positions x and velocities v."""
    N = x.shape[1]
    E_pot = 0.0
    E_kin = 0.0
    # sum up potential energies
    for pair in verlet_list:
        # distance vector
        r_ij = ex_3_4.minimum_image_vector(x[:, pair[0]], x[:, pair[1]], box)
        E_pot += ex_3_4.lj_potential(r_ij, r_cut, shift)
    # sum up kinetic energy
    for i in range(N):
        E_kin += 0.5 * np.dot(v[:, i], v[:, i])
    return E_pot + E_kin


def get_verlet_list(x: np.ndarray, r_cut: float, skin: float, box: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Create a list of interaction partners.

    """
    N = x.shape[1]
    verlet_list = []

    # TODO: YOUR IMPLEMENTATION OF VERLET LISTS GOES HERE...
    for first_particle in range(N):
        for second_particle in range(first_particle + 1, N):
            r_ij = ex_3_4.minimum_image_vector(x[:, second_particle], x[:, first_particle], box)
            if np.linalg.norm(r_ij) < (r_cut + skin):
                verlet_list.append([second_particle, first_particle])

    return np.copy(x), np.array(verlet_list)


def step_vv(x: np.ndarray, v: np.ndarray, f: np.ndarray, dt: float, r_cut: float, skin: float, box: np.ndarray, x0: np.ndarray, verlet_list: np.ndarray):  
    global update_counter
    # update positions
    x += v * dt + 0.5 * f * dt * dt
    # check for maximum distance a particle moved
    max_dx = np.max(np.linalg.norm(x - x0, axis=0))
    if max_dx > 0.5 * skin:
        x0, verlet_list = get_verlet_list(x, r_cut, skin, box)
        update_counter += 1
    # half update of the velocity
    v += 0.5 * f * dt

    # apply pbc
    x, v = ex_3_4.apply_pbc(x, v, box)
    # compute new forces
    f = forces(x, r_cut, box, verlet_list)
    # we assume that all particles have a mass of unity

    # second half update of the velocity
    v += 0.5 * f * dt

    return x, v, f, x0, verlet_list

def init_and_run_simulation(n_per_side: int, skin: float, T_MAX = None) -> Tuple[float, int]:
    """
    write helper
    """
    DT = 0.01
    T_MAX = 1.0 if T_MAX is None else T_MAX
    N_TIME_STEPS = int(T_MAX / DT)

    R_CUT = 2.5
    SHIFT = - 0.016316891136

    SKIN = skin
    nargs=1,

    DIM = 2
    DENSITY = 0.7
    N_PER_SIDE = n_per_side
    N_PART = N_PER_SIDE**DIM
    VOLUME = N_PART / DENSITY
    BOX = np.ones(DIM) * VOLUME**(1. / DIM)

    start_time = time.time()

    # particle positions
    # x = np.array(list(itertools.product(np.linspace(0, BOX[0], N_PER_SIDE, endpoint=False),
    #                                     np.linspace(0, BOX[1], N_PER_SIDE, endpoint=False)))).T
    x = ex_3_5.init_2dgrid_positions(N_PER_SIDE, BOX)

    # random particle velocities
    v = 2.0 * np.random.random((DIM, N_PART)) - 1.0

    x0, verlet_list = get_verlet_list(x, R_CUT, SKIN, BOX)

    f = forces(x, R_CUT, BOX, verlet_list)

    positions = np.zeros((N_TIME_STEPS, DIM, N_PART))
    energies = np.zeros(N_TIME_STEPS)

    for i in tqdm.tqdm(range(N_TIME_STEPS)):
        x, v, f, x0, verlet_list = step_vv(x, v, f, DT, R_CUT, SKIN, BOX, x0, verlet_list) 

        positions[i] = x
        energies[i] = total_energy(x, v, R_CUT, SHIFT, BOX, verlet_list)

    run_time = time.time() - start_time

    print('run_time: ', run_time, 's')
    print('number of updates: ', update_counter)
    
    # plt.plot(energies)
    # plt.show()

    # plt.plot(x[0, :], x[1, :], '.', color='red')
    # plt.axhline(y=0.0, ls=':', color='k')
    # plt.axhline(y=BOX[1], ls=':', color='k')
    # plt.axvline(x=0.0, ls=':', color='k')
    # plt.axvline(x=BOX[0], ls=':', color='k')
    # # plt.savefig('sm1_worksheet_2/plots/grid_positions_5_particles.png', format='png', dpi=600)
    # plt.show()

    return run_time, update_counter


if __name__ == "__main__":
    n_per_side = 8
    skin_values = np.arange(0.0, 1.0, 0.1)

    # plots were made for T_MAX = 40.0
    # run_time_list, counter_list = [], []
    # for skin in skin_values:
    #     run_time, counts = init_and_run_simulation(n_per_side, skin)
    #     run_time_list.append(run_time)
    #     counter_list.append(counts)

    # # generate plots
    # fig, axs = plt.subplots(1, 2,figsize=(12.0, 6.0))

    # axs[0].plot(skin_values, counter_list, marker='o', markersize=6, lw=1, ls=":", color='mediumblue')
    
    # axs[0].set_ylabel("Number of Verlet List Updates")
    # axs[0].set_xlabel('Skin value')
    # axs[0].set_title('Skin value vs. Verlet List Updates')


    # axs[1].plot(skin_values, run_time_list, marker='o', markersize=6, lw=1, ls=":", color='mediumblue')
        
    # axs[1].set_ylabel("Simulation Run Time [s]")
    # axs[1].set_xlabel('Skin value')
    # axs[1].set_title('Skin value vs. Simulation Run Time')

    # fig.tight_layout()
    # # plt.savefig('sm1_worksheet_2/plots/SkinComparison.png', format='png', dpi=600)
    # plt.show()

    # print(run_time_list[2], run_time_list[3])

    ex_3_5_runtimes = np.load('sm1_worksheet_2/plots/ex_3_5_runtimes.npy')
    
    minimal_skin = 0.3
    T_MAX = 2.50
    
    n_per_side_list = np.arange(3, 14, 1, dtype=int)
    simulation_runtimes = []
    for n in n_per_side_list:
        min_run_time, _ = init_and_run_simulation(n, minimal_skin, T_MAX=T_MAX)
        simulation_runtimes.append(min_run_time)


    fig, axs = plt.subplots(1, 2, figsize=(12.0, 6.0))

    axs[0].plot(n_per_side_list**2, simulation_runtimes, marker='o', markersize=6, lw=1, ls=":", color='mediumblue', label='w/ Verlet-List')
    axs[0].plot(n_per_side_list**2, ex_3_5_runtimes, marker='o', markersize=6, lw=1, ls=":", color='orangered', label='w/o Verlet-List')
    axs[0].legend(loc='upper left')

    axs[1].plot(n_per_side_list**2, simulation_runtimes, marker='o', markersize=6, lw=1, ls=":", color='mediumblue')
    axs[1].plot(n_per_side_list**2, ex_3_5_runtimes, marker='o', markersize=6, lw=1, ls=":", color='orangered')
    axs[1].loglog()
    
    axs[0].set_ylabel('Simulation Run Times [s]')
    axs[0].set_xlabel('Particle Number')
    axs[1].set_xlabel('Particle Number')

    axs[0].set_title('vanilla axes')
    axs[1].set_title('double log axes')

    fig.suptitle(f'Number of Particles vs. Simulation runtime for T_MAX={T_MAX}')
    fig.tight_layout()
    plt.savefig('sm1_worksheet_2/plots/OptimalSkinRuntimes.png', format='png', dpi=600)
    plt.show()

