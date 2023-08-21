import numpy as np
from scipy.stats import dirichlet, uniform
from numpy.random import multinomial


def generate_matrix(p, K, N, I, e):
    A = np.zeros((p, K))

    # Generate anchor words
    for k in range(K):
        for i in I[k]:
            A[i, k] = e

    # Generate non-anchor words
    for j in range(p):
        if j not in np.hstack(I):  # skip anchor words
            A[j, :] = uniform.rvs(size=K)

    # Normalize each sub-column
    for k in range(K):
        J = [j for j in range(p) if j not in I[k]]  # non-anchor words for topic k
        A[J, k] = A[J, k] / np.sum(A[J, k]) * (1 - np.sum(A[I[k], k]))

    # Draw columns of W from Dirichlet distribution
    W = dirichlet.rvs([0.3] * K, size=p).T  # Transposed to match the dimensionality

    # Simulate N words from Multinomial
    AW = A.dot(W)
    words = [multinomial(N, AW[:, k]) for k in range(K)]

    return A, W, words


# Example usage
p = 10
K = 3
N = 1000
I = [[0, 1], [2, 3], [4, 5]]  # Example indices of anchor words for each topic
e = K / p
print(generate_matrix(p, K, N, I, e))
