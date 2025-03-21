"""
Fairness Elicitation with No-Regret Learning

Implementation of the paper:
"An Algorithmic Framework for Fairness Elicitation" by Jung et al.
"""

__version__ = "0.1.0"

from .classifier import CostSensitiveClassifier
from .no_regret import NoRegretFairness
from .utils import load_constraints, compute_constraint_weights, load_training_data

__all__ = [
    'CostSensitiveClassifier',
    'NoRegretFairness',
    'load_constraints',
    'compute_constraint_weights',
    'load_training_data',
]