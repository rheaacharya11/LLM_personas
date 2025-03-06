# libraries
import numpy as np
# global variables?
# eta
# A 
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
def best_response(train_set, lambda, tau):
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

def no_regret(train_set, A, C_lambda, C_tau, T, mu_lambda, mu):
    n = len(train_set)

    # parameter initialization
    theta = np.zeros(n, n)
    tau = 0

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
        for i in range(n):
            for j in range(n):
                sum_wa = sum(weights.get((i, j), 0) * alphas[-1][i][j])
                # alphas[-1][i][j]  selects the jth element of the ith element of the last element of alpha
        gradient = sum_wa / len(A) - eta
        tau_t = max(0, min(C_tau, tau + mu_tau))

        D_t, alpha_t = best_response(lambda_t, tau_t)

        classifiers.append(D_t)
        for i in range(n):
            for j in range(n):
                
                
                




        
        