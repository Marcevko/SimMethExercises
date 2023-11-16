import numpy as np
import matplotlib.pyplot as plt

import scipy.linalg


def lj_potential(r_ij: np.ndarray) -> float:
    """
    write helper
    """
    vector_norm = np.linalg.norm(r_ij)
    return 4 * ( (1 / vector_norm**12) - (1 / vector_norm**6) )

def lj_force(r_ij: np.ndarray) -> np.ndarray:
    """
    Write helper
    """
    vector_norm = np.linalg.norm(r_ij)
    return 24 * r_ij * ( (2 / vector_norm**14) - (1 / vector_norm**8) )


if __name__ == "__main__":
    distance_vector = np.zeros((1000, 2))
    distance_vector[:, 0] = np.linspace(0.85, 2.5, 1000)

    lj_potential_computed = np.array([lj_potential(distance) for distance in distance_vector])
    lj_force_computed = np.array([lj_force(distance) for distance in distance_vector])
    lj_potential_cutoff = np.array(
        [lj_potential(distance) + 1 if np.linalg.norm(distance) <= np.power(2, 1/6) else 0.0 for distance in distance_vector]
    )
    
    plt.plot(distance_vector[:, 0], lj_potential_computed, label='LJ-potential')
    plt.plot(distance_vector[:, 0], lj_force_computed[:, 0], label='LJ-force (x-component)')
    plt.plot(distance_vector[:, 0], lj_potential_cutoff, label='shifted/cutted LJ-potential')
    plt.legend()
    plt.ylim(-3.0, 10.0)
    plt.xlabel(r'Distance $d$')
    plt.ylabel(r'$V_{LJ}(d)$ and $F_{LJ, x}(d)$')
    plt.savefig("sm1_worksheet_2/plots/lj_potential_force_plots.png", format='png', dpi=600)
    plt.show()
