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
