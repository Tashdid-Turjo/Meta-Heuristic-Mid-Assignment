# Entropy-Guided Niche Differential Evolution (EG-NDE)

## 1. Problem Description

Many real-world optimization problems are multimodal, i.e., their objective functions contain multiple local and global optima. Classical metaheuristics such as Differential Evolution (DE), Genetic Algorithms (GA), and Particle Swarm Optimization (PSO) tend to converge to a single best solution and lose population diversity, which is undesirable when multiple good solutions are needed.

This project implements:

- A **baseline**: standard Differential Evolution (DE) for continuous minimization.
- A **proposed method**: **Entropy-Guided Niche Differential Evolution (EG-NDE)**, a simplified algorithm inspired by NCD-DE. EG-NDE adds:
  - Greedy niche-center selection using distance thresholds.
  - Niche-wise DE updates in larger niches.
  - Dual-scale local search (wide / narrow Gaussian steps) in very small niches.
  - Entropy-based diversity measurement on the final population.

Both algorithms are benchmarked on multimodal test functions (Rastrigin and Griewank) using:
- Best fitness (mean ± std over 20 runs)
- Mean Shannon entropy of the final population as a diversity indicator.

---

## 2. Dependencies

- Python 3.8+  
- NumPy

Install NumPy with:

```bash
pip install numpy
```

---

## 3. Results

| Function (D=10) | Algorithm | Best Fitness (mean ± std) | Entropy H (mean) | Entropy H (std) |
|----------------|-----------|---------------------------|------------------|-----------------|
| Rastrigin      | DE        | 24.921 ± 6.076            | 1.693            | 0.218           |
| Rastrigin      | EG-NDE    | 32.660 ± 9.081            | 2.209            | 0.169           |
| Griewank       | DE        | 29.177 ± 5.439            | 0.581            | 0.134           |
| Griewank       | EG-NDE    | 158.481 ± 46.290          | 0.663            | 0.069           |
