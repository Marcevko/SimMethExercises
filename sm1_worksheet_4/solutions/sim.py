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


test_uniform_vectors(DT, GAMMA_LANGEVIN)

# used T=k=1
import time 
def step_vv_langevin(x, v, g, dt, gamma):
    # update positions
    x += v * dt * (1 - 0.5* dt * gamma) + 0.5 * g * dt * dt
    # half update of the velocity
    v = (v * (1 - 0.5 * dt * gamma) + 0.5 * dt * g) / (1 + 0.5 * dt * gamma)

    # for this excercise no forces from other particles
    f = np.zeros_like(x)
    g = draw_random_uniform_force(f, dt, gamma)

    # second half update of the velocity
    v += (0.5 * dt * g) / (1 + 0.5 * dt * gamma)
    
    return x, v, g


# SET UP SYSTEM OR LOAD IT
print("Starting simulation...")
t = 0.0
step = 0
measurement_step = 0

# random particle positions
x = np.random.random((N, 3))
v = np.zeros((N, 3))

# variables to cumulate data
ts = []
Es = []
Tms = []
vels = []
traj = []

measurement_num = (TIME_MAX/DT) // MEASUREMENT_STRIDE
traj_arr = np.zeros((int(measurement_num), N, DIM), dtype=float)

# main loop
g = np.zeros_like(x)

print(f"Simulating until tmax={TIME_MAX}...")

while t < TIME_MAX:
    x, v, g = step_vv_langevin(x, v, g, DT, GAMMA_LANGEVIN)

    t += DT
    step += 1

    if step % MEASUREMENT_STRIDE == 0:
        E = compute_energy(v)
        Tm = compute_temperature(v)
        vels.append(v) # changed the flatten
        traj.append(x) # Do not know why, but this list only contains the last position array for ALL timesteps... 

        traj_arr[measurement_step] = x # Therefore the array solution

        # print(f"t={t}, E={E}, T_m={Tm}")

        ts.append(t)
        Es.append(E)
        Tms.append(Tm)

        measurement_step += 1

# at the end of the simulation, write out the final state
datafilename = f'./sm1_worksheet_4/plots/{args.id}.dat.gz'
print(f"Writing simulation data to {datafilename}.")
vels = np.array(vels)
traj = np.array(traj)

datafile = gzip.open(datafilename, 'wb')
pickle.dump([N, T, GAMMA_LANGEVIN, x, v, ts, Es, Tms, vels, traj_arr], datafile) # writing traj_arr now...
datafile.close()

print("Finished simulation.")
print("Computing plots ...")

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

def maxwell_boltzmann(vel: np.ndarray, temp: float):
    prefactor = np.sqrt(2/np.pi)*np.power(1/temp, 3/2)
    exponential = np.exp(- (np.dot(vel, vel)) / (2*temp))
    return prefactor*np.dot(vel, vel)*exponential

# Temperature plot
plt.plot(ts, Tms, label='Temperature trajectory')
plt.plot(ts, running_average(np.asarray(Tms), 100), label='running avetage with M=100')
plt.legend()
plt.xlabel(f'Timesteps')
plt.ylabel(r'Temperature $T$')
plt.savefig('./sm1_worksheet_4/plots/langevin_temperature_plot.png', format='png', dpi=150)
plt.show()

vels_hist = np.linalg.norm(np.asarray(vels), axis=2,)

plot_velocities = np.arange(0.0, 3.0, 0.01)
maxwell_velocities = [np.array([v, 0.0, 0.0]) for v in plot_velocities]
plot_maxwell = [maxwell_boltzmann(vel, T) for vel in maxwell_velocities]
plt.hist(vels_hist[:, 0], bins=25, density=True, label='velocity histogram')
plt.plot(
    plot_velocities, plot_maxwell, label='Maxwell-Boltzmann distribution'
)
plt.legend()
plt.xlabel(r'Velocity $| \bf{v} |$')
plt.ylabel(r'Probability $P(|\bf{v}|)$')
plt.savefig('./sm1_worksheet_4/plots/Maxwell_Boltzmann_distribution.png', format='png', dpi=150)
plt.show()