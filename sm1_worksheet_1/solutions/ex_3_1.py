#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import logging
import itertools
from typing import Tuple, Callable

import scipy.constants

def force(
        r_ij: np.ndarray, 
        m_i: float, 
        m_j: float,
        g: float
    ) -> np.ndarray:
    """
    Computes the gravitational force between two planets.

    Args:
        r_ij: distance vector between particle i and j
        m_i: mass of particle i
        m_i: mass of particle j
        g: gravitational constant
    
    Returns:
        np.ndarray (2, ): force vector between particle pair i-j
    """
    r_ij_abs = np.linalg.norm(r_ij)
    return -g * m_i * m_j * r_ij / (r_ij_abs)**3


def forces_(
        x: np.ndarray,
        masses: np.ndarray,
        g: float,
    ) -> np.ndarray:
    """
    Computes the commulative acting forces on all planets as the sum of all smaller forces from 'force'.    
    
    Args:
        x: Array of position arrays of all 6 particles of shape (2, 6)
        masses: Array of particles masses of shape (6, )
        g: gravitational constant

    Returns:
        np.ndarray (2, 6): array of all cummulative forces acting on all 6 particles.
    """
    force_array = np.zeros(x.shape, dtype=float)

    # generate all possible two-particle-combinations
    particle_pairings = list(
        itertools.combinations(
                np.arange(0, x.shape[1], 1, dtype=int), 2
            )
        )
    
    for pair in particle_pairings:
        first_index, second_index = pair[0], pair[1]
        pair_distance_vector = x[:, first_index] - x[:, second_index]
        pair_force = force(pair_distance_vector, masses[first_index], masses[second_index], g)

        # F_01 = -F_10
        force_array[:, first_index] += pair_force
        force_array[:, second_index] += (-1) * pair_force

    return force_array


def forces(
        x: np.ndarray,
        masses: np.ndarray,
        g: float,
    ) -> np.ndarray:
    """
    Computes the commulative acting forces on all planets as the sum of all smaller forces from 'force'.    
    
    Args:
        x: Array of position arrays of all 6 particles of shape (2, 6)
        masses: Array of particles masses of shape (6, )
        g: gravitational constant

    Returns:
        np.ndarray (2, 6): array of all cummulative forces acting on all 6 particles.
    """
    particles = len(masses)
    forces = np.zeros((2, particles))
    for i in range(particles):
        for j in range(i+1, particles):
            rij = x[:, i] - x[:, j]
            m_i = masses[i]
            m_j = masses[j]
            forces[:, i] += force(rij, m_i, m_j, g)
            forces[:, j] -= force(rij, m_i, m_j, g)

    return forces


def step_euler(
        x: np.ndarray, 
        v: np.ndarray,
        dt: float, 
        masses: np.ndarray,
        gravity: float, 
        force: None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes one integration step with the non-symplectic Euler-Algorithm.

    Args:
        x: Two-dimensional position vector for all planets -> (N, 2, )
        y: Two-dimensional velocity vector for all planets -> (N, 2, )
        dt: time-step size
        mass: masses of all planets -> (N, )
        g: gravitational force constant
        forces: None, not used

    Returns:
        Tuple[np.ndarray, np.ndarray] -> [(N, 2), (N, 2)]: updated positions and velocities of all planets
    """
    a = forces(x, masses, gravity) / masses
    
    x += v * dt
    v += a * dt

    return x, v


def run(
        x0: np.ndarray, 
        v0: np.ndarray,
        dt: float, 
        masses: np.ndarray,
        gravity: float, 
        integrator: Callable,
        number_of_years: int = 1,
    ) -> np.ndarray:
    """
    Simulates the system of planets of a given time-step-size for number_of_years (N/dt steps).

    Args:
        x0: initial set of planet positions -> (N, 2, )
        v0: initial set of planet velocities -> (N, 2, )
        dt: time-step size
        masses: masses of all planets -> (N, )
        gravity: gravitational force constant
        integrator: integration algorithm to be used for simulation
        number_of_years: Duration of simulation in years. Defaults to 1.
        
    Returns:
        np.ndarray: array of position trajectories of all planets
    """
    number_of_steps = number_of_years * int( 1 / dt )

    position_trajectories = np.zeros(
            (number_of_steps + 1, x0.shape[0], v0.shape[1]),
            dtype=np.float32,
        )
    
    x_past, v_past = x0.copy(), v0.copy()
    position_trajectories[0] = x_past

    for step in range(number_of_steps):
        x_t, v_t = integrator(
            x_past,
            v_past,
            dt,
            masses, 
            gravity,
            forces(x_past, masses, gravity),
        )

        position_trajectories[step + 1] = x_t
        x_past, v_past = x_t.copy(), v_t.copy()

    return position_trajectories


if __name__ == "__main__":
    # load the npz file
    try:
        solar_system_data = np.load('files/solar_system.npz')
    except Exception as e:
        raise(e)
        
        
    names = solar_system_data['names']
    x_init = solar_system_data['x_init']
    v_init = solar_system_data['v_init']
    m = solar_system_data['m']
    g = solar_system_data['g']

    time_step_list = [0.001, 0.00075, 0.0005, 0.00025, 0.0001]
    position_trajectories_dict = dict()

    for time_step in time_step_list:
        position_trajectory = run(
            x_init,
            v_init, 
            time_step,
            m,
            g,
            step_euler,
        )
        position_trajectories_dict[time_step] = position_trajectory
    
    # generate plot with all planets for largest time-step
    for planet_index, planet in enumerate(names):
        plt.plot(
            position_trajectories_dict[time_step_list[-1]][:, 0, planet_index],
            position_trajectories_dict[time_step_list[-1]][:, 1, planet_index], 
            lw=3.5, 
            alpha=0.7,
            label=f'{planet}'
            )
    plt.xlabel(f'x in AU')
    plt.ylabel(f'y in AU')
    plt.title(f'Planet trajectories for dt={time_step_list[-1]}')
    plt.legend(loc=(0.60, 0.05))
    plt.grid()
    plt.savefig('plots/trajectories_all_planets.png', format='png', dpi=600)
    plt.show()

    # generate plots for different dt of moon trajectory
    fig, axs = plt.subplots(3, 2, figsize=(12.0, 15.0))
    
    for time_step_index, time_step in enumerate(position_trajectories_dict.keys()):
        moon_trajectory_in_earth_frame = position_trajectories_dict[time_step][:, :, 2] - position_trajectories_dict[time_step][:, :, 1]
        axs[time_step_index // 2, time_step_index % 2].plot(
            moon_trajectory_in_earth_frame[:, 0], 
            moon_trajectory_in_earth_frame[:, 1], 
            lw=3,
            alpha=0.7, 
            label=f'Moon',
            color='cornflowerblue'
            )
        axs[time_step_index // 2, time_step_index % 2].plot([0], [0], 'o', label='Earth', color='mediumseagreen')
        axs[time_step_index // 2, time_step_index % 2].set_title(f'Moon trajectory for dt={time_step}', fontsize=10)

        if time_step_index == 1:
            axs[0, 1].legend(loc='upper right')
        
    fig.suptitle('Comparison of Moon trajectories for different time-step sizes', y=1.0)
    fig.tight_layout()
    plt.savefig('plots/moon_trajectory_comparison.png', format='png', dpi=600)
    plt.show()



    