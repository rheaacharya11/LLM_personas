import numpy as np

def best_response_primal(lambda_matrix, tau, y, n, w, A_size):
    """
    Compute the primal best response given dual variables lambda and tau.
    
    For each example i, we set the cost of predicting 0 or 1 as follows:
      - If y[i] == 0:
            cost0 = 0,
            cost1 = 1/n + sum_j (lambda[i,j] - lambda[j,i])
      - If y[i] == 1:
            cost0 = 1/n,
            cost1 = sum_j (lambda[i,j] - lambda[j,i])
    
    The CSC oracle then returns the classifier h that, for each i, predicts:
            h[i] = 1 if cost1 < cost0, and 0 otherwise.
    
    Also, for every pair (i,j) we set the excess fairness violation indicator:
            alpha[i,j] = 1   if  tau*(w[i,j]/A_size) - lambda_matrix[i,j] <= 0,
                           0   otherwise.
    
    Args:
        lambda_matrix: An (n x n) array of dual weights.
        tau: The dual variable corresponding to the overall fairness constraint.
        y: A length-n array of binary labels.
        n: Number of examples.
        w: An (n x n) fairness weight matrix (e.g. fraction of stakeholders who believe 
           that x_j should be treated at least as well as x_i).
        A_size: The total number of pairs (typically n^2 if all pairs are used).
    
    Returns:
        h: A binary vector of predictions (length n).
        alpha: An (n x n) binary matrix representing the excess fairness violation for each pair.
    """
    cost0 = np.zeros(n)
    cost1 = np.zeros(n)
    for i in range(n):
        # Compute the fairness “bonus” term: sum_{j} (lambda[i,j] - lambda[j,i])
        diff = np.sum(lambda_matrix[i, :] - lambda_matrix[:, i])
        if y[i] == 0:
            cost0[i] = 0.0
            cost1[i] = 1.0/n + diff
        else:  # y[i] == 1
            cost0[i] = 1.0/n
            cost1[i] = diff

    # CSC oracle: for each example, choose the label that minimizes the cost.
    h = np.where(cost1 < cost0, 1, 0)
    
    # Compute fairness violation for each pair (i,j)
    alpha = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if tau * (w[i, j] / A_size) - lambda_matrix[i, j] <= 0:
                alpha[i, j] = 1
            else:
                alpha[i, j] = 0
    return h, alpha

def no_regret_dynamics(y, w, gamma, eta, T, C_lambda, C_tau, mu_lambda, mu_tau):
    """
    Run the no-regret dynamics (primal-dual algorithm) for fairness-constrained empirical risk minimization.
    
    This function simulates T rounds of the following dynamics:
      1. Compute lambda from the internal dual variable theta:
             lambda = C_lambda * exp(theta) / (1 + sum(exp(theta))).
      2. Update tau by a gradient step (projected to [0, C_tau]):
             tau ← projection_{[0, C_tau]} [tau + mu_tau * ((1/|A|) * sum_{i,j} w[i,j]*alpha_prev[i,j] - eta)].
      3. Compute the primal best response (via best_response_primal) to get classifier h and matrix alpha.
      4. Update theta (for each (i,j)) by:
             theta[i,j] ← theta[i,j] + mu_lambda * ( (h[i] - h[j]) - alpha[i,j] - gamma ).
    
    Args:
        y: Array of labels (length n, with values 0 or 1).
        w: Fairness weight matrix of shape (n, n).
        gamma: Fairness margin parameter.
        eta: Overall fairness violation budget.
        T: Number of iterations (rounds).
        C_lambda: Bound for the dual variable lambda.
        C_tau: Bound for the dual variable tau.
        mu_lambda: Step size for updating theta (and thus lambda).
        mu_tau: Step size for updating tau.
    
    Returns:
        h_avg: The averaged (over T rounds) classifier predictions (an array of length n).
        h_list: List of classifier predictions (each of length n) from each round.
        alpha_list: List of alpha matrices (each shape (n, n)) from each round.
    """
    n = len(y)
    A_size = n * n  # If we use all pairs in S x S.
    theta = np.zeros((n, n))  # Internal dual variable for lambda.
    tau = 0.0
    h_list = []
    alpha_list = []
    alpha_prev = np.zeros((n, n))
    
    for t in range(T):
        # Compute lambda from theta.
        exp_theta = np.exp(theta)
        sum_exp = np.sum(exp_theta)
        lambda_matrix = C_lambda * exp_theta / (1 + sum_exp)
        
        # Update tau with gradient step and project to [0, C_tau].
        tau = tau + mu_tau * ((np.sum(w * alpha_prev) / A_size) - eta)
        tau = min(max(tau, 0.0), C_tau)
        
        # Primal best response: get classifier h and fairness violation matrix alpha.
        h, alpha = best_response_primal(lambda_matrix, tau, y, n, w, A_size)
        h_list.append(h)
        alpha_list.append(alpha)
        
        # Update theta: for each pair (i, j)
        for i in range(n):
            for j in range(n):
                theta[i, j] = theta[i, j] + mu_lambda * ((h[i] - h[j]) - alpha[i, j] - gamma)
        
        # Save alpha for use in the next tau update.
        alpha_prev = alpha.copy()
    
    # Form the final classifier as the average over the T rounds.
    h_avg = np.mean(np.array(h_list), axis=0)
    return h_avg, h_list, alpha_list

# === Example usage ===
if __name__ == "__main__":
    # For demonstration, we simulate a small dataset.
    n = 10  # number of examples
    np.random.seed(0)
    # Random binary labels.
    y = np.random.randint(0, 2, size=n)
    # In practice, the weight matrix w is derived from stakeholder pairwise elicitation.
    # Here we simply set w to be all ones (i.e. every pair is equally important).
    w = np.ones((n, n))
    
    # Set hyperparameters.
    gamma = 0.1      # fairness margin
    eta = 0.05       # overall fairness violation budget
    T = 100          # number of rounds
    C_lambda = 1.0   # dual variable bound for lambda
    C_tau = 1.0      # dual variable bound for tau
    mu_lambda = 0.1  # step size for lambda updates
    mu_tau = 0.1     # step size for tau updates
    
    # Run the no-regret dynamics algorithm.
    h_avg, h_list, alpha_list = no_regret_dynamics(y, w, gamma, eta, T, C_lambda, C_tau, mu_lambda, mu_tau)
    
    print("Final average predictions (per example):")
    print(h_avg)
