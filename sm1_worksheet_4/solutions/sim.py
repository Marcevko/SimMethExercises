import argparse
import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt

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


parser = argparse.ArgumentParser()
parser.add_argument('id', type=int, help='Simulation id')
args = parser.parse_args()

def compute_energy(v):
    return (v * v).sum() / 2.

def compute_temperature(v):
    e_kin = compute_energy(v)

    temp = 2*e_kin / (N * DIM)
    return temp


def step_vv(x, v, f, dt):
    # update positions
    x += v * dt + 0.5 * f * dt * dt

    # half update of the velocity
    v += 0.5 * f * dt

    # for this excercise no forces from other particles
    f = np.zeros_like(x)

    # second half update of the velocity
    v += 0.5 * f * dt

    return x, v, f

def draw_random_force_(f, dt, gamma):
    random_vector = np.random.normal(0.0, np.sqrt(2*T*gamma/dt), f.shape)
    return f + random_vector

def draw_random_force(f: np.ndarray, dt: float, gamma: float) -> np.ndarray:
    wanted_std = np.sqrt(2*gamma*T/dt)

    random_vector = np.random.random(f.shape)
    random_vector -= np.mean(random_vector, axis=0)
    random_vector *= wanted_std / np.std(random_vector, axis=0)

    assert random_vector.shape == f.shape
    assert np.allclose(np.mean(random_vector, axis=0), np.zeros(3))
    assert np.allclose(np.std(random_vector, axis=0), np.full((3), wanted_std))

    # plt.hist(random_vector, bins=20)
    # plt.show()

    return f + random_vector

# def test_random_numbers(gamma, dt):
#     wanted_std = np.sqrt(2*gamma*T/dt)

#     test_arr = np.random.uniform(size=1000)
#     test_arr -= np.mean(test_arr, axis=0)
#     plt.hist(test_arr, bins=20)
#     plt.show()
#     test_arr *= wanted_std / np.std(test_arr, axis=0)

#     plt.hist(test_arr, bins=20)
#     plt.show()

#     print(np.mean(test_arr))
#     print(np.std(test_arr), wanted_std)

# test_random_numbers(GAMMA_LANGEVIN, DT)

# used T=k=1
def step_vv_langevin(x, v, f, dt, gamma):
    # draw random force G
    g = draw_random_force_(f, dt, gamma)    

    # update positions
    x += v * dt * (1 - 0.5* dt * gamma) + 0.5 * g * dt * dt

    # half update of the velocity
    v = v * (1 - 0.5*dt*gamma) + 0.5 * dt * g
    v /= (1 + 0.5* dt * gamma) 

    # for this excercise no forces from other particles
    f = np.zeros_like(x)
    g = draw_random_force_(f, dt, gamma)

    # second half update of the velocity
    v = v * (1 - 0.5*dt*gamma) + 0.5 * dt * g
    v /= (1 + 0.5* dt * gamma) 
    
    return x, v, f


# SET UP SYSTEM OR LOAD IT
print("Starting simulation...")
t = 0.0
step = 0

# random particle positions
x = np.random.random((N, 3))
v = np.zeros((N, 3))

# variables to cumulate data
ts = []
Es = []
Tms = []
vels = []
vels_ = []
traj = []


# main loop
f = np.zeros_like(x)


print(f"Simulating until tmax={TIME_MAX}...")

while t < TIME_MAX:
    x, v, f = step_vv_langevin(x, v, f, DT, GAMMA_LANGEVIN)

    t += DT
    step += 1

    if step % MEASUREMENT_STRIDE == 0:
        E = compute_energy(v)
        Tm = compute_temperature(v)
        vels.append(v.flatten())
        traj.append(x.flatten())
        vels_.append(v)
        # print(f"t={t}, E={E}, T_m={Tm}")

        ts.append(t)
        Es.append(E)
        Tms.append(Tm)


# at the end of the simulation, write out the final state
datafilename = f'{args.id}.dat.gz'
print(f"Writing simulation data to {datafilename}.")
vels = np.array(vels)
traj = np.array(traj)

datafile = gzip.open(datafilename, 'wb')
pickle.dump([N, T, GAMMA_LANGEVIN, x, v, ts, Es, Tms, vels, traj], datafile)
datafile.close()

print("Finished simulation.")
print("Computing plots ...")

plt.plot(ts, Tms)
plt.show()

def compute_quadratically_averaged_velocities(vels: np.ndarray) -> np.ndarray:
    """
    vels: velocity vectors of all particles. shape: (timesteps, particles, dimensions)
    """
    averaged_vector = np.linalg.norm(
                vels, axis=2,
    )
    return averaged_vector

def maxwell_boltzmann(vel: np.ndarray, temp: float):
    prefactor = 4*np.pi*np.power(2*np.pi*temp, 2/3)
    exponential = np.exp(- (np.dot(vel, vel)) / (2*temp))
    return prefactor*np.dot(vel, vel)*exponential

vels_hist = compute_quadratically_averaged_velocities(np.asarray(vels_))

plot_velocities = np.arange(0.0, 4.0, 0.01)
maxwell_velocities = [np.array([v, 0.0, 0.0]) for v in plot_velocities]
plot_maxwell = [maxwell_boltzmann(vel, T) for vel in maxwell_velocities]
plt.hist(vels_hist[:, 0], bins=20, density=True)
plt.plot(
    plot_velocities, plot_maxwell, label='was passiert hier'
)
plt.legend()
plt.show()