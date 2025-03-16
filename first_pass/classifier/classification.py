# libraries
import numpy as np
# global variables?
# eta
# A 
alphas = {}
for i in range(n):
    for j in range(n):
        if (i, j) in constraint_pairs:  
            alphas[(i, j)] = 0.0
A = len(alphas)
# use to get alpha_ij  alpha_ij = alphas.get((i, j), 0.0)


# Weights
weights = {}
for i in range(n):
    for j in range(n):
        if (i, j) in constraint_pairs:
            weights[(i, j)] = calculate_weight(i, j)
# Cost-Sensitive Oracle
costs = []

def csc(train_set, costs):
    D = []*n
    for i in range(n):
        if costs[i][0] < costs[i][1]:
            D[i] = 0
        else:
            D[i] = 1

# Algorithm One
def best_response(train_set, lambda, tau, alpha):
    n = len(train_set)
    
    for i in range(n):
        # lambda ij - lambda ji 
        lambda_sum = 0
        for j in range(n):
            lambda_sum += lambda[i][j] - lambda[j][i]

        # set costs
        if train_set[i][observed] == 0:
            costs[i][0] = 0
            costs[i][1] = 1/n + lambda_sum
        else:
            costs[i][0] = 1/n
            costs[i][1] = lambda_sum

    D = csc(train_set, costs)
    for i in range(n):
        for j in range(n):
            value = (tau * w[i][j] / A) - lambda[i][j]
            if value <= 0:
                alphas[(i, j)] = 0
            else:
                alphas[(i, j)] = 1
    return D, alphas


def no_regret(train_set, A, C_lambda, C_tau, T, mu_lambda, mu):
    n = len(train_set)

    # parameter initialization
    theta = np.zeros((n, n))
    tau = 0

    # store classifers and alphas
    classifiers = []
    all_alphas = []

    for t in range (0, T):
        # initialize lambda t matrix for this round
        lambda_t = np.zeros(n, n)

        # denominator is same for all pairs
        denominator = 1
        for i in range(n):
            for j in range(n):
                denominator += np.exp(theta[i][j])
        # calculate specific lambda_t value
        for i in range(n):
            for j in range(n):
                lambda_t[i][j] = C_lambda * np.exp(theta[i][j])/ denominator

        # set tau_t

        # calculate weighted sum of alpha's
        sum_wa = 0
        for (i, j), alpha_val in alphas.items():
            if (i, j) in weights:
                sum_wa += weights[(i, j)] * alpha_val

        gradient = sum_wa / len(A) - eta
        tau_t = max(0, min(C_tau, tau + mu_tau * gradient))

        D_t, alpha_t = best_response(train_set, lambda_t, tau_t, alphas)

        classifiers.append(D_t)
        all_alphas.append(alpha_t)
        for i in range(n):
            for j in range(n):
                prediction_diff = D_t[i] - D_t[j]

                theta[i][j] = theta[i][j] + mu_lambda * (prediction_diff - alpha_t.get((i, j) - gamma)

                
                




        
        