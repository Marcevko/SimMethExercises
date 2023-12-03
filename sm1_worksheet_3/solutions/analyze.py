import pickle
import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('file', help="Path to pickle file.")
args = parser.parse_args()

with open(args.file, 'rb') as fp:
    data = pickle.load(fp)

print(data.keys())
plt.plot(data['energies'])
plt.show()

def running_average(O, M):
    pass
