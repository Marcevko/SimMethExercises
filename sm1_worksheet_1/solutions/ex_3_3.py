#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import logging

import ex_3_1
import ex_3_2

if __name__ == "__main__":
    # load the npz file
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
        'non-symplectic Euler': ex_3_1.run(x_init, v_init, time_step, m, g, ex_3_1.step_euler, number_of_years=10),
        'symplectic Euler': ex_3_1.run(x_init, v_init, time_step, m, g, ex_3_2.step_symplectic_euler, number_of_years=10),
        'velocity-verlet': ex_3_1.run(x_init, v_init, time_step, m, g, ex_3_2.step_velocity_verlet, number_of_years=10),
    }

    # generate comparison plot
    fig, axs = plt.subplots(2, 3, figsize=(18.0, 5.0))

    for integrator_index, (integrator, trajectory) in enumerate(position_trajectories_dict.items()): 
        moon_trajectory_in_earth_frame = trajectory[:, :, 2] - trajectory[:, :, 1]
        axs[0, integrator_index].plot(
            moon_trajectory_in_earth_frame[:, 0],
            moon_trajectory_in_earth_frame[:, 1],
            lw=3,
            alpha=0.7, 
            label=f'Moon',
            color='cornflowerblue',
        )
        axs[0, integrator_index].plot([0], [0], 'o', label='Earth', color='mediumseagreen')
        axs[0, integrator_index].set_title(f'{integrator} Algorithm')

        distance_trajectory = np.linalg.norm(moon_trajectory_in_earth_frame, axis=1)

        axs[1, integrator_index].plot(
            distance_trajectory, 
            color='red',
            lw=2,
            label='Distance Moon-Earth',
        )

        if integrator_index == 0:
            axs[0, integrator_index].legend(loc='upper right')
            axs[1, integrator_index].legend(loc='upper left')


    fig.suptitle(f'Comparison of Moon-Earth distance for dt={time_step}')        
    fig.tight_layout()
    plt.savefig('plots/integrator_distance_comparison.png', format='png', dpi=600)
    plt.show()

    for integrator_index, (integrator, trajectory) in enumerate(position_trajectories_dict.items()): 
        moon_trajectory_in_earth_frame = trajectory[:, :, 2] - trajectory[:, :, 1]
        
        distance_trajectory = np.linalg.norm(moon_trajectory_in_earth_frame, axis=1)

        plt.plot(
            distance_trajectory, 
            alpha=0.85, 
            lw=3,
            label=f'{integrator}',
        )
        plt.title(f'Moon-Earth distances for different integrators')
        plt.legend(loc='upper left')
        plt.xlabel(f'integration steps')
        plt.ylabel(f'Distance in AU')

    plt.savefig('plots/integrator_distances_comparison_one_plot.png', format='png', dpi=600)
    plt.show()    
        