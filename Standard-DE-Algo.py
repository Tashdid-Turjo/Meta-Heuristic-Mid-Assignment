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
