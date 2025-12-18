# Python + NumPy implementation of standard DE/rand/1/bin for minimization (DE function Code):

import numpy as np

def de(
    func,
    bounds,
    dim,
    pop_size=30,
    F=0.5,
    CR=0.9,
    max_gens=200,
    seed=None,
):
    """
    Standard Differential Evolution (DE/rand/1/bin) for minimization.

    Parameters
    ----------
    func : callable
        Objective function f(x), where x is a 1D numpy array of length dim.
    bounds : list of (low, high)
        Variable bounds, length = dim. Example: [(-5.12, 5.12)] * dim
    dim : int
        Problem dimension.
    pop_size : int
        Population size (NP).
    F : float
        Differential weight (typically 0.4–0.9).
    CR : float
        Crossover rate (0–1).
    max_gens : int
        Number of generations.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    best_x : np.ndarray
        Best solution found.
    best_f : float
        Best objective value.
    best_history : list[float]
        Best fitness after each generation.
    population : np.ndarray
        Final population (pop_size x dim).
    """

    rng = np.random.default_rng(seed)

    # 1) Initialize population uniformly within bounds
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        low, high = bounds[d]
        pop[:, d] = rng.uniform(low, high, size=pop_size)

    # Evaluate initial population
    fitness = np.array([func(ind) for ind in pop])

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    best_history = [best_f]

