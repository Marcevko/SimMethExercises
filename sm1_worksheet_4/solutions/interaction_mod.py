import argparse
import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def SC_lattice(N: int, box: np.ndarray) -> np.ndarray:
    """Get an array of positions for N particles in a simple-cubic
        lattice.

        Parameters
        ----------
        N : int
            Number of particles.
        box : (3, 0) array_like
            Simulation box dimentions.

        Returns
        -------
        x : (N, 3) array_like
            Particle positions.

    """
    # Calculate the lattice spacing based on the number of particles and box dimensions
    lattice_spacing = (box / np.power(N, 1/3)).astype(float)

    # Generate particle positions in a simple cubic lattice
    positions = []
    for indx_x in range(N):
        for indx_y in range(N):
            for indx_z in range(N):
                positions.append([indx_x * lattice_spacing[0], indx_y * lattice_spacing[1], indx_z * lattice_spacing[2]])

    # Convert the list of positions to a NumPy array
    x = np.array(positions[:N])
    return x
                

def minimum_image_vector(r1: np.ndarray, r2: np.ndarray, box: np.ndarray):
        """ Get connecting vector of two particles, respecting minimum image convention.

        Parameters
        ----------
        r1 : (3, ) array_like
            Position of particle 1.
        r2 : (3, ) array_like
            Position of particle 2.
        box : (3, ) array_like
            Simulation box dimensions.

        Returns
        -------
        array_like
            Connecting vector for `r1` and `r2` with minimum image convention.

        """
        r_12 = r2 - r1
        r_12 -= np.rint(r_12 / box) * box
        return r_12

def get_verlet_list(x: np.ndarray, box: np.ndarray, r_cut: int, skin: int):
        """Create a list of index pairs of interacting particles.

        Parameters
        ----------
        x : (N, 3) array_like
            Particle positions.
        box : (3, 0) array_like
            Simulation box dimentions.
        r_cut : int
            Cutoff radius.
        skin : int
            Skin (Verlet list).

        Returns
        -------
        x : (N, 3) array_like
            Particle positions at the creation of Verlet list.
        array_like : (, 2)
            Verlet list.
        """
        N = x.shape[0]
        verlet_list = []

        for i in range(1, N):
            for j in range(i):
                r_ij = minimum_image_vector(x[i, :], x[j, :], box)

                if np.linalg.norm(r_ij) < r_cut + skin:
                    verlet_list.append((i, j))

        return np.copy(x), np.array(verlet_list)

def lj_force(r_ij: np.ndarray, r_cut: int):
    """Get forces for two LJ-interacting particles.

        Parameters
        ----------
        r_ij : (3) array_like
            Distance vector between particles i and j.
        r_cut : int
            Cutoff radius.

        Returns
        -------
        array_like : (3)
            Force vector.
    """
    def _lj_force(r_ij):
        r = np.linalg.norm(r_ij)
        fac = 4.0 * (12.0 * np.power(r, -13.) - 6.0 * np.power(r, -7.))
        return fac * r_ij / r

    r = np.linalg.norm(r_ij)
    if r < r_cut:
        return _lj_force(r_ij)
    return np.zeros_like(r_ij)

def lj_potential(r_ij: np.ndarray, r_cut: int, shift: int):
    """Get LJ potential between two particles, with 
    epsilon = 1 and sigma =1.

        Parameters
        ----------
        r_ij : (3) array_like
            Distance vector between particles i and j.
        r_cut : int
            Cutoff radius.
        shift : int
            shift of the potential.

        Returns
        -------
        int
            Potential value.
    """
    def _lj_potential(r_ij):
        r = np.linalg.norm(r_ij)
        return 4.0 * (np.power(r, -12.) - np.power(r, -6.))

    r = np.linalg.norm(r_ij)
    if r < r_cut:
        return _lj_potential(r_ij) + shift
    return 0.0

def forces(x: np.ndarray, r_cut: int, verlet_list: np.ndarray, box: np.ndarray):
    """Compute and return the forces acting on the particles,
    depending on the positions x.
    
        Parameters
        ----------
        x : (N, 3) array_like
            Particle positions.
        r_cut : int
            Cutoff radius.
        verlet_list : (N, 2) array_like
            Verlet list.
        box : (3, 0) array_like
            Simulation box dimentions.

        Returns
        -------
        array_like : (N, 3)
            Forces.
    """
    N = x.shape[0]
    f = np.zeros_like(x)

    print(x[0, :].shape)

    for (i, j) in verlet_list:
        r_ij = minimum_image_vector(x[i, :], x[j, :], box)
        lj_force_ij = lj_force(r_ij, r_cut)
        f[i, :] += lj_force_ij
        f[j, :] -= lj_force_ij

    return f

# SYSTEM CONSTANTS
# timestep
DT = 0.01
# length of run
TIME_MAX = 2000.0
# desired temperature
T = 0.3
# total number of particles
N = 50
DIM = 3
# friction coefficient
GAMMA_LANGEVIN = 0.8
# number of steps to do before the next measurement
MEASUREMENT_STRIDE = 50
# Parameters for WCA potential
R_CUT = np.power(2, 1/6)
SHIFT = 1.0

SKIN = 0.3

parser = argparse.ArgumentParser()
parser.add_argument('id', type=int, help='Simulation id')
parser.add_argument('density', type=float, help='Density of the system')
args = parser.parse_args()

VOLUME = N / args.density
BOX = np.full(3, np.power(VOLUME, 1/3))

def compute_kin_energy(v: np.ndarray):
    return (v * v).sum() / 2.

def compute_pot_energy(x: np.ndarray, r_cut: float, shift: float, verlet_list: list, box: np.ndarray):
    potential_energy = 0.0
    for (i, j) in verlet_list:
        r_ij = minimum_image_vector(x[i, :], x[j, :], box)
        lj_potential_ij = lj_potential(r_ij, r_cut, shift)
        potential_energy += lj_potential_ij
    
    return potential_energy
    

def compute_total_energy(x: np.ndarray, v: np.ndarray, r_cut: float, shift: float, verlet_list: list, box: np.ndarray):
    return compute_kin_energy(v) + compute_pot_energy(x, r_cut, shift, verlet_list, box)

def compute_temperature(v):
    e_kin = compute_kin_energy(v)

    temp = 2*e_kin / (N * DIM)
    return temp


def step_vv(x, v, f, dt):
    # update positions
    x += v * dt + 0.5 * f * dt * dt

    # half update of the velocity
    v += 0.5 * f * dt

    # for this excercise no forces from other particles
    f = forces(x, R_CUT, SHIFT, BOX)

    # second half update of the velocity
    v += 0.5 * f * dt

    return x, v, f

def draw_random_gaussian_force(f, dt, gamma):
    random_vector = np.random.normal(0.0, np.sqrt(2*T*gamma/dt), f.shape)
    return f + random_vector

def draw_random_uniform_force(f: np.ndarray, dt: float, gamma: float) -> np.ndarray:
    wanted_std = np.sqrt(2*gamma*T/dt)

    random_vector = np.random.uniform(low=-np.sqrt(3)*wanted_std, high=np.sqrt(3)*wanted_std, size=f.shape)
    return f + random_vector

def test_uniform_vectors(dt, gamma):
    random_test_vector = draw_random_uniform_force(np.zeros((1000000, 3)), dt, gamma)

    assert np.allclose(np.mean(random_test_vector, axis=0), np.zeros(3), atol=0.05)
    assert np.allclose(np.std(random_test_vector, axis=0), np.full((3), np.sqrt(2*gamma*T/dt)), atol=0.05)


# used T=k=1
def step_vv_langevin(x, v, g, dt, gamma, x0, r_cut, box, skin, verlet_list):
    # update positions
    x += v * dt * (1 - 0.5* dt * gamma) + 0.5 * g * dt * dt
    # half update of the velocity
    v = (v * (1 - 0.5 * dt * gamma) + 0.5 * dt * g) / (1 + 0.5 * dt * gamma)

    max_dx = np.max(np.linalg.norm(x - x0, axis=0))
    if max_dx > 0.5 * SKIN:
        x0, verlet_list = get_verlet_list(x, box, r_cut, skin)

    # for this excercise no forces from other particles
    f = forces(x, r_cut, verlet_list, box)
    g = draw_random_uniform_force(f, dt, gamma)

    # second half update of the velocity
    v += (0.5 * dt * g) / (1 + 0.5 * dt * gamma)
    
    return x, v, g, x0, verlet_list


# SET UP SYSTEM OR LOAD IT
print("Starting simulation...")
t = 0.0
step = 0
measurement_step = 0

# random particle positions
x = SC_lattice(N, BOX)
v = np.zeros((N, 3))
x0, verlet_list = get_verlet_list(x, BOX, R_CUT, SKIN)

# variables to cumulate data
ts = []
Es = []
Tms = []
vels = []
traj = []

measurement_num = (TIME_MAX/DT) // MEASUREMENT_STRIDE
traj_arr = np.zeros((int(measurement_num), N, DIM), dtype=float)

# main loop
f = forces(x, R_CUT, verlet_list, BOX)
g = draw_random_uniform_force(f, DT, GAMMA_LANGEVIN)

print(f"Simulating until tmax={TIME_MAX}...")

while t < TIME_MAX:
    x, v, g, x0, verlet_list = step_vv_langevin(x, v, g, DT, GAMMA_LANGEVIN, x0, R_CUT, BOX, SKIN, verlet_list)

    t += DT
    step += 1

    if step % MEASUREMENT_STRIDE == 0:
        E = compute_total_energy(x, v, R_CUT, SHIFT, verlet_list, BOX)
        Tm = compute_temperature(v)
        vels.append(v) # changed the flatten
        traj.append(x) # Do not know why, but this list only contains the last position array for ALL timesteps... 

        traj_arr[measurement_step] = x # Therefore the array solution

        ts.append(t)
        Es.append(E)
        Tms.append(Tm)

        measurement_step += 1
        print('measurement step: ', measurement_step)

    

# at the end of the simulation, write out the final state
datafilename = f'./sm1_worksheet_4/plots/{args.id}.dat.gz'
print(f"Writing simulation data to {datafilename}.")
vels = np.array(vels)
traj = np.array(traj)

datafile = gzip.open(datafilename, 'wb')
pickle.dump([N, T, GAMMA_LANGEVIN, x, v, ts, Es, Tms, vels, traj_arr], datafile) # writing traj_arr now...
datafile.close()

print("Finished simulation.")