import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

import os
from typing import Callable


def f(x: float) -> float:
    term1 = -2 * x**2 * np.sin(x) * np.cos(x) - 2 * x * (np.sin(x))**2
    term2 = np.exp(-x**2 * (np.sin(x))**2)

    result = term1 * term2
    return result


def simple_sampling(f: Callable, a: float, b: float, N: int) -> float:
    """
    Computes the approximate integral of a function f in an interval [a, b] with N steps.
    """
    uniform_x_sites = np.linspace(a, b, N)
    integral_value = ( (b-a) / N) * f(uniform_x_sites).sum()

    return integral_value

if __name__=="__main__":

    integration_interval = (0.1, 10.0)

    # plot f(x) in the Interval [0.1, 50.0]
    plot_interval = np.arange(0.1, 50.0, 0.01)

    plt.plot(plot_interval, f(plot_interval), color='brown', label=f'f(x)',)
    plt.xlabel(f'x')
    plt.ylabel(f'f(x)')
    # plt.grid()
    plt.legend()
    plt.savefig('./sm1_worksheet_5/plots/ex_1_funcplot.png', format='png', dpi=150)
    plt.show()

    # compute exact solution of integral
    x = sp.symbols('x')

    term1 = -2 * x**2 * sp.sin(x) * sp.cos(x) - 2 * x * (sp.sin(x))**2
    term2 = sp.exp(-x**2 * (sp.sin(x))**2)
    f_x = term1 * term2

    exact_integral = sp.integrate(f_x, x)
    definite_integral = sp.integrate(f_x, (x, integration_interval[0], integration_interval[1]))

    print(f"Exact integral of f(x): {exact_integral}")
    print(f"Definite integral of f(x): {definite_integral}")

    # evaluate simple_sampling
    N_list = [2**i for i in np.arange(2, 21, 1)]
    simple_sampling_results = [
        simple_sampling(f, integration_interval[0], integration_interval[1], n) for n in N_list
    ]
    simple_sampling_error = [np.abs(result - definite_integral) for result in simple_sampling_results]

    plt.plot(N_list, simple_sampling_error, '.', label=f'actual error')
    plt.xscale('log')
    plt.xlabel(r'number of integration sites $N$')
    plt.ylabel(f'Integration error')
    plt.legend()
    plt.show()

    """
    TODO:
    statistical error mal nachsehen und hier implementieren. Wird wahrscheinlich in der VL definiert sein
    """

