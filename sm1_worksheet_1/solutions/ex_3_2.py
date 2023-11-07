#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import logging
from typing import Tuple

import ex_3_1

def step_symplectic_euler(
        x: np.ndarray,
        v: np.ndarray, 
        dt: float, 
        mass: np.ndarray, 
        g: float, 
        forces: None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes one integration step with the symplectic Euler-Algorithm.

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
    a = ex_3_1.forces(x, mass, g) / mass
    
    v += a * dt
    x += v * dt

    return x, v

def step_velocity_verlet(
        x: np.ndarray,
        v: np.ndarray, 
        dt: float, 
        mass: np.ndarray, 
        g: float,
        force_old: None
    ):
    """
    Computes one integration step with the symplectic Velocity-Verlet-Algorithm.

    Args:
        x: Two-dimensional position vector for all planets -> (N, 2, )
        y: Two-dimensional velocity vector for all planets -> (N, 2, )
        dt: time-step size
        mass: masses of all planets -> (N, )
        g: gravitational force constant
        force_old: None, not used

    Returns:
        updated positions and velocities of all planets: Tuple[np.ndarray, np.ndarray] -> [(N, 2), (N, 2)]
    """
    a = ex_3_1.forces(x, mass, g) / mass
    
    x += v * dt + 0.5 * a * dt**2
    v += 0.5 * a * dt
    
    a_t = ex_3_1.forces(x, mass, g) / mass
    v += 0.5 * a_t * dt

    return x, v


if __name__ == "__main__":

    try:
        solar_system_data = np.load('files/solar_system.npz')
    except Exception as e:
        logging.warning(e)
        # implement general way of loading the path
        # os.chdir oder sys.path.append
        
    names = solar_system_data['names']
    x_init = solar_system_data['x_init']
    v_init = solar_system_data['v_init']
    m = solar_system_data['m']
    g = solar_system_data['g']

    time_step = 0.01
    position_trajectories_dict = {
        'non-symplectic Euler': ex_3_1.run(x_init, v_init, time_step, m, g, ex_3_1.step_euler),
        'symplectic Euler': ex_3_1.run(x_init, v_init, time_step, m, g, step_symplectic_euler),
        'velocity-verlet': ex_3_1.run(x_init, v_init, time_step, m, g, step_velocity_verlet),
    }

    # generate comparison plot
    fig, axs = plt.subplots(1, 3, figsize=(18.0, 5.0))

    for integrator_index, (integrator, trajectory) in enumerate(position_trajectories_dict.items()): 
        moon_trajectory_in_earth_frame = trajectory[:, :, 2] - trajectory[:, :, 1]
        axs[integrator_index].plot(
            moon_trajectory_in_earth_frame[:, 0],
            moon_trajectory_in_earth_frame[:, 1],
            lw=3,
            alpha=0.7, 
            label=f'Moon',
            color='cornflowerblue',
        )
        axs[integrator_index].plot([0], [0], 'o', label='Earth', color='mediumseagreen')
        
        if integrator_index == 2:
            axs[integrator_index].legend(loc='upper left')

        axs[integrator_index].set_title(f'{integrator} Algorithm')

    fig.suptitle(f'Comparison of integrator algorithms for dt={time_step}')        
    fig.tight_layout()
    plt.savefig('plots/integrator_comparison.png', format='png', dpi=600)
    plt.show()

    
