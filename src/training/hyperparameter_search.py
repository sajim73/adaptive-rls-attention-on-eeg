"""
hyperparameter_search.py (work-in-progress)

Grid/random search utilities for RLS model hyperparameters (lambda, forgetting factor, etc.).
Replace placeholders with your notebook-extracted routines.
"""

from typing import Dict, Any, List
import itertools

def grid(param_grid: Dict[str, List[Any]]):
    keys = list(param_grid.keys())
    for vals in itertools.product(*[param_grid[k] for k in keys]):
        yield dict(zip(keys, vals))

def hyperparameter_search_single_subject(run_fn, param_grid):
    best, best_score = None, float("-inf")
    for params in grid(param_grid):
        score = run_fn(**params)
        if score > best_score:
            best, best_score = params, score
    return best, best_score
