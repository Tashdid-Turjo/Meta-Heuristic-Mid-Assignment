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
    Standard Differential Evolution (DE/rand/1/bin) for minimization. Here, bin -> binomial crossover.

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
    best_idx : int
        Index of the best individual in the population (position of the minimum value in fitness).
    pop : np.ndarray
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

    for gen in range(max_gens):
        for i in range(pop_size):
            # 2) Mutation: DE/rand/1
            # choose 3 distinct indices different from i
            idxs = np.arange(pop_size)
            idxs = idxs[idxs != i]
            r1, r2, r3 = rng.choice(idxs, size=3, replace=False)
            x1, x2, x3 = pop[r1], pop[r2], pop[r3]
            v = x1 + F * (x2 - x3)  # mutant vector

            # 3) Crossover: binomial
            u = pop[i].copy()
            j_rand = rng.integers(0, dim)  # ensure at least one dimension from v
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    u[j] = v[j]

            # 4) Bound handling (simple clipping)
            for d in range(dim):
                low, high = bounds[d]
                if u[d] < low:
                    u[d] = low
                elif u[d] > high:
                    u[d] = high

            # 5) Selection
            f_u = func(u)
            if f_u < fitness[i]:
                pop[i] = u
                fitness[i] = f_u

        # Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_f:
            best_f = fitness[best_idx]
            best_x = pop[best_idx].copy()

        best_history.append(best_f)

    return best_x, best_f, best_history, pop






#######################################################################
# 1.1 DE algo with Rastrigin Function (Entropy H{mean}):
def population_entropy_1d(pop, bounds, bins=20):
    
    """
    Shannon entropy of population distribution along first dimension.

    pop    : array of shape (pop_size, dim)
    bounds : list of (low, high) per dimension
    bins   : number of histogram bins
    """
    x = pop[:, 0]  # use first dimension
    low, high = bounds[0]

    # Histogram over [low, high]
    counts, _ = np.histogram(x, bins=bins, range=(low, high))

    total = counts.sum()
    if total == 0:
        return 0.0

    p = counts / total
    # Avoid log(0): mask zero entries
    p_nonzero = p[p > 0]

    H = -np.sum(p_nonzero * np.log(p_nonzero))  # natural log
    return H





#######################################################################
# 1.2 DE algo with Rastrigin Function (Best Fitness):
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


dim = 10
bounds = [(-5.12, 5.12)] * dim

best_values = []
entropy_values = []

# Will run for 20 times.
for run in range(20):                       
    best_x, best_f, hist, final_pop = de(
        func=rastrigin,
        bounds=bounds,
        dim=dim,
        pop_size=30,
        F=0.5,
        CR=0.9,
        max_gens=300,
        seed= 41 + run,
    )

    best_values.append(best_f)
    
    # Entropy of final population for this run
    H = population_entropy_1d(final_pop, bounds, bins=20)
    entropy_values.append(H)


print("\nEach best fitness value for Rastrigin Function:")
for val in best_values:
    print(repr(val))

print("\nEach entropy H value for Rastrigin Function:")
for H in entropy_values:
    print(repr(H))
    
# For (mean ± std):
mean_best = np.mean(best_values)
std_best = np.std(best_values)

# Avg of 20 values -> mean
print("\nMean best fitness for Rastrigin Function:", mean_best)

print("Std of best fitness for Rastrigin Function:", std_best)

# Mean entropy H
mean_entropy = np.mean(entropy_values)
std_entropy = np.std(entropy_values)

print("Mean entropy H for Rastrigin Function:", mean_entropy)
print("Std of entropy H for Rastrigin Function:", std_entropy)





#######################################################################
# 2.1 DE algo with Griewan Function (Entropy H{mean}):
def pop_entropy_1d(pop, bounds, bins=20):
    
    """
    Shannon entropy of population distribution along first dimension.

    pop    : array of shape (pop_size, dim)
    bounds : list of (low, high) per dimension
    bins   : number of histogram bins
    """
    x = pop[:, 0]  # use first dimension
    low, high = bounds[0]

    # Histogram over [low, high]
    counts, _ = np.histogram(x, bins=bins, range=(low, high))

    total = counts.sum()
    if total == 0:
        return 0.0

    p = counts / total
    # Avoid log(0): mask zero entries
    p_nonzero = p[p > 0]

    H = -np.sum(p_nonzero * np.log(p_nonzero))  # natural log
    return H





#######################################################################
# 2.2 DE algo with Griewan Function (Best Fitness):
def griewan(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


dim = 10
bounds = [(-600.0, 600.0)] * dim   # standard Griewank range

best_values = []
entropy_values = []

# Will run for 20 times.
for run in range(20):                       
    best_x, best_f, hist, final_pop = de(
        func=griewan,
        bounds=bounds,
        dim=dim,
        pop_size=30,
        F=0.5,
        CR=0.9,
        max_gens=300,
        seed= 41 + run,
    )

    best_values.append(best_f)
    
    # Entropy of final population for this run
    H = pop_entropy_1d(final_pop, bounds, bins=20)
    entropy_values.append(H)


print("\nEach best fitness value for Griewan Function:")
for val in best_values:
    print(repr(val))

print("\nEach entropy H value for Griewan Function:")
for H in entropy_values:
    print(repr(H))
    
# For (mean ± std):
mean_best = np.mean(best_values)
std_best = np.std(best_values)

# Avg of 20 values -> mean
print("\nMean best fitness for Griewan Function:", mean_best)

print("Std of best fitness for Griewan Function:", std_best)

# Mean entropy H
mean_entropy = np.mean(entropy_values)
std_entropy = np.std(entropy_values)

print("Mean entropy H for Griewan Function:", mean_entropy)
print("Std of entropy H for Griewan Function:", std_entropy)