import numpy as np

def eg_nde(
    func,
    bounds,
    dim,
    pop_size=30,
    F=0.5,
    CR=0.9,
    max_gens=300,
    d_min=1.0,      # minimum distance between niche centers
    seed=None,
):
    """
    Entropy-Guided Niche Differential Evolution (EG-NDE)
    Simplified NCD-DE-style algorithm.

    - Uses DE/rand/1/bin inside each niche (if niche size >= 3)
    - Uses dual-scale local search when niche size < 3
    - Niche centers chosen by greedy distance threshold (no internal GA)
    """

    rng = np.random.default_rng(seed)

    # --- init population ---
    pop = np.zeros((pop_size, dim))
    for d in range(dim):
        low, high = bounds[d]
        pop[:, d] = rng.uniform(low, high, size=pop_size)

    # Initialize population uniformly within bounds
    fitness = np.array([func(ind) for ind in pop])

    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    best_history = [best_f]

    def clip_to_bounds(x):
        for d in range(dim):
            low, high = bounds[d]
            if x[d] < low:
                x[d] = low
            elif x[d] > high:
                x[d] = high
        return x

    for gen in range(max_gens):

        # ---------- Step 1: choose niche centers (greedy NCD) ----------
        idx_sorted = np.argsort(fitness)  # best first
        centers = []

        for idx in idx_sorted:
            if len(centers) == 0:
                centers.append(idx)
            else:
                # distance to existing centers
                dists = [
                    np.linalg.norm(pop[idx] - pop[c])
                    for c in centers
                ]
                if np.min(dists) > d_min:
                    centers.append(idx)

        # ---------- Step 2: partition population into niches ----------
        # niches: dict center_idx -> list of member indices
        niches = {c: [] for c in centers}
        for i in range(pop_size):
            # find nearest center
            dists = [np.linalg.norm(pop[i] - pop[c]) for c in centers]
            nearest_center = centers[int(np.argmin(dists))]
            niches[nearest_center].append(i)

        new_pop = pop.copy()
        new_fit = fitness.copy()

        # ---------- Step 3: update each niche ----------
        for center_idx, members in niches.items():
            if len(members) >= 3:
                # --- DE/rand/1/bin inside niche (niching mutation) ---
                for i in members:
                    # mutation indices from same niche, excluding i if possible
                    cand = [m for m in members if m != i]
                    if len(cand) < 3:
                        continue
                    r1, r2, r3 = rng.choice(cand, size=3, replace=False)
                    x1, x2, x3 = pop[r1], pop[r2], pop[r3]
                    v = x1 + F * (x2 - x3)

                    # crossover
                    u = pop[i].copy()
                    j_rand = rng.integers(0, dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            u[j] = v[j]
                    u = clip_to_bounds(u)

                    f_u = func(u)
                    if f_u < new_fit[i]:
                        new_pop[i] = u
                        new_fit[i] = f_u

            else:
                # --- dual-scale local search for small niches (size 1 or 2) ---
                for i in members:
                    # nearest neighbor in whole population (exclude self)
                    cand = [k for k in range(pop_size) if k != i]
                    dists = [np.linalg.norm(pop[i] - pop[k]) for k in cand]
                    nn = cand[int(np.argmin(dists))]
                    x_i = pop[i]
                    x_nn = pop[nn]

                    if rng.random() <= 0.5:
                        # wide-scale
                        dist = np.linalg.norm(x_i - x_nn)
                        u = x_i + rng.normal(0.0, 1.0, size=dim) * dist
                    else:
                        # narrow-scale
                        direction = x_nn - x_i
                        u = x_i + 0.5 * rng.normal(0.0, 1.0, size=dim) * direction

                    u = clip_to_bounds(u)
                    f_u = func(u)
                    if f_u < new_fit[i]:
                        new_pop[i] = u
                        new_fit[i] = f_u

        # commit updates
        pop = new_pop
        fitness = new_fit

        # update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_f:
            best_f = fitness[best_idx]
            best_x = pop[best_idx].copy()

        best_history.append(best_f)

    return best_x, best_f, best_history, pop
    




#######################################################################
# 1.2 EG-NDE algo with Rastrigin Function (Best Fitness):
# Rastrigin objective
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

dim = 10
bounds = [(-5.12, 5.12)] * dim

best_vals = []
entropy_vals = []

# Will run for 20 times.
for run in range(20):
    best_x, best_f, hist, final_pop = eg_nde(
        func=rastrigin,
        bounds=bounds,
        dim=dim,
        pop_size=30,
        F=0.5,
        CR=0.9,
        max_gens=300,
        d_min=1.0,
        seed=41 + run,
    )
    
    best_vals.append(best_f)
    
    # Entropy of final population for this run
    H = pop_entropy_1d(final_pop, bounds, bins=20)
    entropy_vals.append(H)

print("Rastrigin EG-NDE mean best:", np.mean(best_vals))
print("Rastrigin EG-NDE std best :", np.std(best_vals))
print("Rastrigin EG-NDE mean H   :", np.mean(entropy_vals))
print("Rastrigin EG-NDE std H    :", np.std(entropy_vals))
