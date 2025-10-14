# Simple GP using Quantale-based variation operators
# This code will run in the notebook environment and display results to the user.
# It implements a commutative quantale Q = [0,1] with join = max, product = multiplication.
# Programs map contexts X (discrete indices) -> Q values. Crossover and mutation are pointwise quantale ops.

import random, math, statistics
from dataclasses import dataclass, field
from typing import Dict, List, Callable
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Quantale implementation (concrete choice: Q = [0,1]) ---
class Quantale:
    """A simple commutative quantale on [0,1] with join=max and product=multiply."""
    def __init__(self):
        pass
    def join(self, a: float, b: float) -> float:
        return max(a,b)
    def product(self, a: float, b: float) -> float:
        return a * b
    def unit(self) -> float:
        return 1.0
    def bottom(self) -> float:
        return 0.0

Q = Quantale()

# --- Program representation ---
# A program is a mapping X -> Q where X is {0,..,n-1}.
@dataclass
class Program:
    mapping: Dict[int, float]  # values in [0,1]
    def copy(self):
        return Program(dict(self.mapping))

# Utility to evaluate distance-based fitness against a target
def fitness(program: Program, target: Dict[int,float]) -> float:
    # Higher is better: use negative L1 distance scaled to [0,1]
    xs = sorted(target.keys())
    dist = sum(abs(program.mapping[x] - target[x]) for x in xs)
    maxdist = len(xs) * 1.0
    return 1.0 - (dist / maxdist)  # 1 is perfect match, 0 is worst

# Variation operators using quantale operations (pointwise)
def join_crossover(p1: Program, p2: Program) -> Program:
    mapping = {x: Q.join(p1.mapping[x], p2.mapping[x]) for x in p1.mapping}
    return Program(mapping)

def product_crossover(p1: Program, p2: Program) -> Program:
    mapping = {x: Q.product(p1.mapping[x], p2.mapping[x]) for x in p1.mapping}
    return Program(mapping)

# Two mutation styles as quantale operations:
# - join-mutation: pointwise join with a small random element (makes values weaker/higher)
# - product-mutation: pointwise product with a random element in [0,1] (shrinks values)
def join_mutation(program: Program, intensity=0.2) -> Program:
    mapping = {}
    for x,v in program.mapping.items():
        delta = random.random() * intensity  # in [0,intensity)
        mapping[x] = Q.join(v, min(1.0, delta))  # raise towards delta (join increases value)
    return Program(mapping)

def product_mutation(program: Program, intensity=0.2) -> Program:
    mapping = {}
    for x,v in program.mapping.items():
        factor = 1.0 - (random.random() * intensity)  # in (1-intensity, 1]
        mapping[x] = Q.product(v, factor)
    return Program(mapping)

# Helper: random program
def random_program(n):
    return Program({i: random.random() for i in range(n)})

# --- Simple GP loop ---
def run_gp(
    Xsize=10,
    pop_size=50,
    generations=60,
    crossover_type='join',  # 'join' or 'product'
    mutation_type='join',   # 'join' or 'product'
    tournament_k=3,
    target=None,
    seed=1
):
    random.seed(seed)
    np.random.seed(seed)
    # create target if not provided
    if target is None:
        # for demo, create a target with structure (e.g. a peaked shape)
        target = {i: math.exp(-(i - (Xsize-1)/2)**2 / (2*((Xsize/4)**2))) for i in range(Xsize)}
        # normalize to [0,1] by dividing by max
        mx = max(target.values())
        target = {i: v/mx for i,v in target.items()}
    # initialize population
    pop = [random_program(Xsize) for _ in range(pop_size)]
    history = []
    best_prog = None
    best_fit = -1.0

    for gen in range(generations):
        # evaluate
        fits = [fitness(p, target) for p in pop]
        # record stats
        avg_fit = sum(fits)/len(fits)
        gen_best_idx = max(range(len(pop)), key=lambda i: fits[i])
        gen_best = pop[gen_best_idx]
        gen_best_fit = fits[gen_best_idx]
        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_prog = gen_best.copy()
        history.append({'gen': gen, 'avg_fit': avg_fit, 'best_fit': gen_best_fit})

        # produce next generation
        newpop = []
        while len(newpop) < pop_size:
            # tournament selection for parents
            def tournament():
                cand = random.sample(list(range(pop_size)), tournament_k)
                return max(cand, key=lambda i: fits[i])
            p1 = pop[tournament()]
            p2 = pop[tournament()]
            # crossover
            if crossover_type == 'join':
                child = join_crossover(p1, p2)
            else:
                child = product_crossover(p1, p2)
            # mutation
            if mutation_type == 'join':
                child = join_mutation(child, intensity=0.2)
            else:
                child = product_mutation(child, intensity=0.2)
            newpop.append(child)
        pop = newpop

    # final evaluation
    fits = [fitness(p, target) for p in pop]
    final_best_idx = max(range(len(pop)), key=lambda i: fits[i])
    final_best = pop[final_best_idx]
    final_best_fit = fits[final_best_idx]
    history_df = pd.DataFrame(history)
    return {
        'target': target,
        'best': best_prog,
        'best_fit': best_fit,
        'final_best': final_best,
        'final_best_fit': final_best_fit,
        'history': history_df
    }

# Run experiments with both crossover semantics and compare
res_join_join = run_gp(crossover_type='join', mutation_type='join', seed=2)
res_prod_prod = run_gp(crossover_type='product', mutation_type='product', seed=2)

# Display summary statistics
summary = pd.DataFrame([
    {'variant': 'join-cross & join-mutate', 'best_fit': res_join_join['best_fit'], 'final_best_fit': res_join_join['final_best_fit']},
    {'variant': 'prod-cross & prod-mutate', 'best_fit': res_prod_prod['best_fit'], 'final_best_fit': res_prod_prod['final_best_fit']},
])
import caas_jupyter_tools as jt; jt.display_dataframe_to_user("Quantale GP summary", summary)

# Show fitness over generations for one run
plt.figure(figsize=(8,4))
plt.plot(res_join_join['history']['gen'], res_join_join['history']['best_fit'], label='join best')
plt.plot(res_join_join['history']['gen'], res_join_join['history']['avg_fit'], label='join avg')
plt.plot(res_prod_prod['history']['gen'], res_prod_prod['history']['best_fit'], label='prod best')
plt.plot(res_prod_prod['history']['gen'], res_prod_prod['history']['avg_fit'], label='prod avg')
plt.xlabel('generation')
plt.ylabel('fitness (higher better)')
plt.title('Quantale GP: fitness over generations')
plt.legend()
plt.grid(True)
plt.show()

# Print best program mappings (rounded)
def show_program(name, prog):
    m = {k: round(v,3) for k,v in prog.mapping.items()}
    print(f"--- {name} ---")
    print(m)

show_program('Join-run best (checkpoint)', res_join_join['best'])
print('best_fit (checkpoint):', round(res_join_join['best_fit'],4))
show_program('Join-run final best', res_join_join['final_best'])
print('final_best_fit:', round(res_join_join['final_best_fit'],4))
print()
show_program('Prod-run best (checkpoint)', res_prod_prod['best'])
print('best_fit (checkpoint):', round(res_prod_prod['best_fit'],4))
show_program('Prod-run final best', res_prod_prod['final_best'])
print('final_best_fit:', round(res_prod_prod['final_best_fit'],4))

# Also display target
print("\nTarget mapping (rounded):")
print({k: round(v,3) for k,v in res_join_join['target'].items()})

# Provide history dataframe to user
jt.display_dataframe_to_user("Join-run history", res_join_join['history'])


