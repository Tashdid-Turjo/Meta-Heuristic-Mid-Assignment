Algorithm 1: Entropy-Guided Niche Differential Evolution (EG-NDE)

Initialize population P of NP individuals uniformly within bounds; evaluate f(x) for all x ∈ P.
for gen = 1 to MaxGen do
    Sort P by fitness and greedily select niche centers with pairwise distance > d_min.
    Assign each individual to its nearest center, forming niches S1, …, SK.
    for each niche Sk do
        if |Sk| ≥ 3 then
            For each i ∈ Sk, apply DE/rand/1 mutation and binomial crossover using parents from Sk,
            then perform parent–offspring selection (keep better of trial and parent).
        else
            For each i ∈ Sk, choose wide-scale or narrow-scale Gaussian local search at random,
            then perform parent–offspring selection (keep better of trial and parent).
        end if
    end for
    Update global best solution in P.
end for
Return best solution and final population (later used to compute entropy H for diversity analysis).