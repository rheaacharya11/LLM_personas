#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.linear_model import LogisticRegression
import json 
from tqdm import tqdm 
import scipy.sparse as sp



class FairnessData:
    def __init__(self, X, Y, constraints, weights):
        self.X = sp.csr_matrix(X)
        self.Y = np.array(Y)
        self.constrains = constraints

    def get_constraints_for(self, x):
        return self.constraints.get(x, [])

    def get_weight
## PRIMAL PLAYER

A = pairs w/ constraings
def best_response():
    for i in range(n):
        for j in range(n):
            calculation = (tau * weights[(i, j)] / len(A)) - lambda_values














# initialize lagrangian multipliers
lambda_values = np.zeros(len(constraints))
tau = 0.0
alpha = np.zeros(len(constraints))

# tracking varaibles
models = []
errors_history = []
violations_history = []

def train(X, y, constraints, weights, gamma = 0.0, eta = 0.1, T=100):
    for t in range(T):
        # step 1: cost sensitive classification costs
        costs = create_costs(X, y, constraints, lambda_values)

        # step 2: best response classifier
        h_t = oracle.best_response(X, costs)
        models.append(h_t)

        # step 3: compute fairness violations
        violations = []
        for (i, j) in constraints:
            violation = max(0, h_t[i] - h_t[j] - alpha[idx] - gamma)



