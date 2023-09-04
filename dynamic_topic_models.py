import numpy as np
from scipy.stats import dirichlet
from numpy.linalg import norm
from score_functions import score_W
from preprocessing import corpus_matrix, read_vocab, read_docs


def simulation_plan(c=1 / 4, K=5, n_range=(20, 50), T_range=(20, 50), p=1000):
    results = {}

    docs = read_docs('data/press_data.txt')  # Replace with your document file path
    vocab = read_vocab('data/vocabulary.txt')  # Replace with your vocabulary file path
    D = corpus_matrix(docs, vocab)

    for n in range(n_range[0], n_range[1] + 1):
        for T in range(T_range[0], T_range[1] + 1):
            # Initialize
            A = np.random.rand(p, K)  # A matrix
            theta = np.random.dirichlet(np.ones(K) * 2, n).T  # K x n
            W = np.random.rand(K, n)  # W

            # Simulate sequence W up to 2T to reach stationary regime
            for t in range(2 * T):
                Delta_t = dirichlet.rvs(theta.T)  # n x K
                W = (1 - c) * W + c * Delta_t.T

            # Calculate Pi
            Pi = A @ W  # p x n

            result = score_W(6, 9, 14, D)

            A_hat_r = result.rx2('A_hat')
            W_hat_r = result.rx2('W_hat')

            # Convert the R matrix to numpy array
            A_hat = np.array(A_hat_r)
            W_hat = np.array(W_hat_r)

            # Estimate theta (Step 3)
            hat_theta = np.mean(W, axis=1)

            # Estimate c (Step 3)
            bar_w = np.mean(W[:, :-1], axis=1)
            bar_w_plus = np.mean(W[:, 1:], axis=1)
            numerator = np.sum(np.inner(W[:, :-1] - bar_w[:, np.newaxis], W[:, 1:] - bar_w_plus[:, np.newaxis]))
            denominator = np.sum(norm(W - bar_w[:, np.newaxis], axis=0) ** 2)
            hat_c = 1 - numerator / denominator

            # Estimate ||theta_j|| (Step 3)
            var_W = np.var(W, axis=1)
            hat_norm_theta = (hat_c / (2 - hat_c)) * np.sum(
                np.diag(hat_theta) - np.outer(hat_theta, hat_theta)) / np.sum(var_W)

            # Store results
            results[(n, T)] = {
                'hat_theta': hat_theta,
                'hat_c': hat_c,
                'hat_norm_theta': hat_norm_theta
            }

    return results
