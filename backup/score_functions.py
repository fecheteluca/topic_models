import numpy as np
import cvxpy as cp
import itertools
from scipy.linalg import solve
from matplotlib import pyplot as plt
from scipy.sparse.linalg import svds, eigs
from sklearn.cluster import KMeans
from scipy.optimize import nnls
from numpy.linalg import eig


def score(K, K0, m, D, scatterplot=False):
    """
        Calculates the scores for the given dataset.

        Parameters:
        K (int): The number of singular values to compute.
        K0 (int): The number of clusters to be computed.
        m (int): The number of clusters to be computed.
        D (ndarray): Input data matrix.
        scatterplot (bool): Flag to display a scatterplot, default is False.

        Returns:
        A_hat (ndarray): Adjusted matrix.
        R (ndarray): Matrix after applying singluar value decomposition.
        V (ndarray): Vertices of the computed clusters.
        Pi (ndarray): The computed matrix Pi.
        theta (ndarray): Cluster centers computed by KMeans clustering.
    """
    p = D.shape[0]
    Xi, _, _ = svds(D, K)

    # Step 1
    Xi[:, 0] = abs(Xi[:, 0])
    R = Xi[:, 1:K] / Xi[:, 0, np.newaxis]

    # Step 2
    V, theta = vertices_est(R, K0, m)

    if scatterplot:
        import matplotlib.pyplot as plt
        plt.scatter(R[:, 0], R[:, 1])
        plt.scatter(V[:, 0], V[:, 1], color='r')
        plt.show()

    # Step 3
    Pi = np.linalg.lstsq(np.hstack((V, np.ones((V.shape[0], 1)))), np.hstack((R, np.ones((p, 1)))), rcond=None)[0]
    Pi = np.clip(Pi, 0, None)
    Pi /= np.sum(Pi, axis=0)

    # Step 4
    A_hat = Xi[:, 0, np.newaxis] * Pi

    # Step 5
    A_hat /= np.sum(A_hat, axis=0)

    return A_hat, R, V, Pi, theta


def vertices_est(R, K0, m):
    """
        Estimates the vertices for the given dataset.

        Parameters:
        R (ndarray): The matrix after applying singluar value decomposition.
        K0 (int): The number of clusters to be computed.
        m (int): The number of clusters to be computed.

        Returns:
        theta (ndarray): Selected vertices for the dataset.
        theta_original (ndarray): Original vertices of the dataset.
    """
    K = R.shape[1] + 1

    # Step 2a
    kmeans = KMeans(n_clusters=m, max_iter=K*100, n_init=K*10).fit(R)
    theta = kmeans.cluster_centers_
    theta_original = theta.copy()

    # Step 2b'
    inner = np.dot(theta, theta.T)
    distance = np.diag(inner)[:, np.newaxis] + np.diag(inner) - 2 * inner
    top2 = np.unravel_index(np.argmax(distance), distance.shape)
    theta0 = theta[top2, :].reshape(-1, theta.shape[1])
    theta = np.delete(theta, top2, axis=0)

    if K0 > 2:
        for k0 in range(3, K0+1):
            inner = np.dot(theta, theta.T)
            distance = np.ones((k0-1, 1)) @ np.diag(inner)[np.newaxis, :] - 2 * np.dot(theta0, theta.T)
            ave_dist = np.mean(distance, axis=0)
            index = np.argmax(ave_dist)
            theta0 = np.vstack((theta0, theta[index, :]))
            theta = np.delete(theta, index, axis=0)
        theta = theta0

    # Step 2b
    from itertools import combinations
    comb = list(combinations(range(K0), K))
    max_values = [0] * len(comb)
    for i in range(len(comb)):
        for j in range(K0):
            max_values[i] = max(simplex_dist(theta[j, :], theta[np.array(comb[i]), :]), max_values[i])

    min_index = np.argmin(max_values)

    return theta[np.array(comb[min_index]), :], theta_original


def simplex_dist(theta, V):
    """
        Computes the distance of a point to a simplex.

        Parameters:
        theta (ndarray): The point for which the distance is to be computed.
        V (ndarray): Vertices of the simplex.

        Returns:
        float: The computed distance.
    """
    VV = np.dot(np.hstack((np.eye(V.shape[0]-1), -np.ones((V.shape[0]-1, 1)))), V)
    D = np.dot(VV, VV.T)
    d = np.dot(VV, theta - V[-1, :])

    x = cp.Variable(len(d))
    constraints = [x >= 0, cp.sum(x) == 1]

    # Defining the objective. Note that cvxpy minimizes by default.
    objective = cp.Minimize(cp.quad_form(x, D) + 2 * d @ x)

    # Defining the problem.
    problem = cp.Problem(objective, constraints)

    # Solving the problem.
    min_value = problem.solve()

    return np.sum(np.square(theta - V[-1, :])) + 2 * min_value


def norm_score_N(K, K0, m, N, D, scatterplot=False):
    """
        Normalizes the scores for the given dataset using given parameters.

        Parameters:
        K (int): The number of singular values to compute.
        K0 (int): The number of clusters to be computed.
        m (int): The number of clusters to be computed.
        N (ndarray): Input matrix.
        D (ndarray): Input data matrix.
        scatterplot (bool): Flag to display a scatterplot, default is False.

        Returns:
        A_hat (ndarray): Adjusted matrix.
        R (ndarray): Matrix after applying singluar value decomposition.
        V (ndarray): Vertices of the computed clusters.
        theta (ndarray): Cluster centers computed by KMeans clustering.
    """
    p, n = D.shape
    M = np.sum(D / np.tile(N, (p, 1)), axis=0)
    u, s, _ = svds(D * np.sqrt(1 / M), K)

    # Step 1
    u[:, 0] = np.abs(u[:, 0])
    R = u[:, 1:K] / u[:, 0][:, None]

    # Step 2
    vertices_est_obj = vertices_est(R, K0, m)
    V = vertices_est_obj['V']
    theta = vertices_est_obj['theta']

    if scatterplot:
        print("scatterplot option is not implemented in this Python function")

    # Step 3
    Pi = np.dot(np.column_stack((R, np.ones(p))), solve(np.row_stack((V, np.ones(K))), np.eye(K)))
    Pi = np.maximum(Pi, 0)
    temp = np.sum(Pi, axis=1)
    Pi = (Pi.T / temp).T

    # Step 4
    A_hat = np.sqrt(M) * u[:, 0][:, None] * Pi

    # Step 5
    temp = np.sum(A_hat, axis=0)
    A_hat = (A_hat.T / temp).T

    return A_hat, R, V, theta


def norm_score(K, K0, m, D, Mquantile=0, scatterplot=False, VHMethod='SVS'):
    """
        Computes normalized scores based on input data matrix and method.

        Parameters:
        K (int): The number of singular values to compute.
        K0 (int): The number of vertices in the polytope.
        m (int): The number of vertices for the VHMethod.
        D (np.ndarray): The input data matrix.
        Mquantile (float): The quantile for computation of M_trunk. Default is 0.
        scatterplot (bool): Whether to plot scatterplot. Default is False.
        VHMethod (str): The method for computation, can be 'SVS', 'SP' or 'SVS-SP'. Default is 'SVS'.

        Returns:
        Tuple: Tuple containing A_hat (estimated matrix), R (the normalized scores), V (the computed vertices),
        Pi (computed proportions) and theta (cluster centers).
    """
    p = D.shape[0]
    M = np.mean(D, axis=1)
    M_trunk = np.minimum(M, np.quantile(M, Mquantile))

    obj = svds(np.sqrt(np.reciprocal(M_trunk))[:, None] * D, K)
    Xi = obj[0]

    # Step 1
    Xi[:, 0] = np.abs(Xi[:, 0])
    R = np.apply_along_axis(lambda x: x / Xi[:, 0], 0, Xi[:, 1:K])

    # Step 2
    if VHMethod == 'SVS':
        vertices_est_obj = vertices_est(R, K0, m)
        V = vertices_est_obj['V']
        theta = vertices_est_obj['theta']
    elif VHMethod == 'SP':
        vertices_est_obj = successiveProj(R, K)
        V = vertices_est_obj['V']
        theta = None
    elif VHMethod == 'SVS-SP':
        vertices_est_obj = vertices_est_SP(R, m)
        V = vertices_est_obj['V']
        theta = None

    if scatterplot:
        print('wrong!')
        plt.scatter(R[:, 0], R[:, 1])
        plt.scatter(V[:, 0], V[:, 1], color='red')
        plt.show()

    # Step 3
    Pi = np.hstack((R, np.ones((p, 1)))) @ solve(np.vstack((V, np.ones((1, K)))), np.eye(K + 1))
    Pi = np.maximum(Pi, 0)
    temp = np.sum(Pi, axis=0)
    Pi = np.apply_along_axis(lambda x: x / temp, 0, Pi)

    # Step 4
    A_hat = np.sqrt(M_trunk)[:, None] * Xi[:, 0][:, None] * Pi

    # Step 5
    temp = np.sum(A_hat, axis=1)
    A_hat = np.apply_along_axis(lambda x: x / temp, 0, A_hat.T)

    return A_hat, R, V, Pi, theta


def score_W(K, K0, m, D):
    """
       Calculates the scores for W.

       Parameters:
       K (int): The number of singular values to compute.
       K0 (int): The number of vertices in the polytope.
       m (int): The number of vertices.
       D (np.ndarray): The input data matrix.

       Returns:
       Tuple: Tuple containing A_hat (estimated matrix), W_hat (estimated matrix), R (the normalized scores),
       V (the computed vertices) and Pi (computed proportions).
    """
    p = D.shape[0]
    obj = svds(D, k=K)
    Xi = obj[0]

    # Step 1
    Xi[:, 0] = np.abs(Xi[:, 0])
    R = np.apply_along_axis(lambda x: x / Xi[:, 0], 0, Xi[:, 1:K])

    # Step 2
    vertices_est_obj = vertices_est(R, K0, m)
    V = vertices_est_obj['V']

    # Step 3
    n = R.shape[0]
    Pi = np.hstack((R, np.ones((n, 1)))) @ solve(np.vstack((V, np.ones((1, K)))), np.eye(K + 1))
    Pi = np.maximum(Pi, 0)
    temp = np.sum(Pi, axis=0)
    Pi = np.apply_along_axis(lambda x: x / temp, 0, Pi)

    # Step 4
    W_tilde = Pi * Xi[:, 0]
    W_tilde_norm = np.apply_along_axis(lambda x: x - np.mean(x), 0, W_tilde)
    Q_direction = eig(W_tilde_norm.T @ W_tilde_norm)[1][:, K - 1]

    # Step 5
    W_hat = np.apply_along_axis(lambda x: x / np.sum(x), 0, W_tilde * np.abs(Q_direction))

    # Step 6: nnls to recover A
    A_hat = np.zeros((p, K))
    for i in range(p):
        A_hat[i, :] = nnls(W_hat.T, D[i, :])[0]
    for k in range(K):
        A_hat[:, k] /= np.sum(A_hat[:, k])

    return A_hat, W_hat, R, V, Pi


def debias_score_N(K, K0, m, N, D, scatterplot=False):
    """
        Calculates the debiased score for N.

        Parameters:
        K (int): The number of singular values to compute.
        K0 (int): The number of vertices in the polytope.
        m (int): The number of vertices.
        N (np.ndarray): The normalization matrix.
        D (np.ndarray): The input data matrix.
        scatterplot (bool): Whether to plot scatterplot. Default is False.

        Returns:
        Tuple: Tuple containing A_hat (estimated matrix), R (the normalized scores), V (the computed vertices),
        Pi (computed proportions) and theta (cluster centers).
    """
    p = D.shape[0]
    n = D.shape[1]
    M = np.sum(D / np.tile(N, (p, 1)).T, axis=0)

    obj = eigs(D @ D.T - np.diag(M), K)
    Xi = obj[1]

    # Step 1
    Xi[:, 0] = np.abs(Xi[:, 0])
    R = np.apply_along_axis(lambda x: x / Xi[:, 0], 0, Xi[:, 1:K])

    # Step 2
    vertices_est_obj = vertices_est(R, K0, m)
    V = vertices_est_obj['V']
    theta = vertices_est_obj['theta']

    if scatterplot:
        plt.scatter(R[:, 0], R[:, 1])
        plt.scatter(V[:, 0], V[:, 1], color='r')
        plt.show()

    # Step 3
    Pi = np.hstack((R, np.ones((p, 1)))) @ solve(np.vstack((V, np.ones((1, K)))), np.eye(K + 1))
    Pi = np.maximum(Pi, np.zeros(Pi.shape))
    temp = Pi.sum(axis=0)
    Pi = np.apply_along_axis(lambda x: x / temp, 0, Pi)

    # Step 4
    A_hat = Xi[:, 0][:, np.newaxis] * Pi

    # Step 5
    temp = A_hat.sum(axis=1)
    A_hat = np.apply_along_axis(lambda x: x / temp, 0, A_hat).T

    return A_hat, R, V, Pi, theta


def debias_score(K, K0, m, N, D, scatterplot=False):
    """
       Calculates the debiased score.

       Parameters:
       K (int): The number of singular values to compute.
       K0 (int): The number of vertices in the polytope.
       m (int): The number of vertices.
       N (np.ndarray): The normalization matrix.
       D (np.ndarray): The input data matrix.
       scatterplot (bool): Whether to plot scatterplot. Default is False.

       Returns:
       Tuple: Tuple containing A_hat (estimated matrix), R (the normalized scores), V (the computed vertices),
       Pi (computed proportions) and theta (cluster centers).
    """
    p = D.shape[0]
    n = D.shape[1]
    M = D.mean(axis=1)

    obj = eigs(D @ D.T - (n / np.mean(N)) * np.diag(M), K)
    Xi = obj[1]

    # Step 1
    Xi[:, 0] = np.abs(Xi[:, 0])
    R = np.apply_along_axis(lambda x: x / Xi[:, 0], 0, Xi[:, 1:K])

    # Step 2
    vertices_est_obj = vertices_est(R, K0, m)
    V = vertices_est_obj['V']
    theta = vertices_est_obj['theta']

    if scatterplot:
        plt.scatter(R[:, 0], R[:, 1])
        plt.scatter(V[:, 0], V[:, 1], color='r')
        plt.show()

    # Step 3
    Pi = np.hstack((R, np.ones((p, 1)))) @ solve(np.vstack((V, np.ones((1, K)))), np.eye(K + 1))
    Pi = np.maximum(Pi, np.zeros(Pi.shape))
    temp = Pi.sum(axis=0)
    Pi = np.apply_along_axis(lambda x: x / temp, 0, Pi)

    # Step 4
    A_hat = Xi[:, 0][:, np.newaxis] * Pi

    # Step 5
    temp = A_hat.sum(axis=1)
    A_hat = np.apply_along_axis(lambda x: x / temp, 0, A_hat).T

    return A_hat, R, V, Pi, theta


def error1_A(A, A_hat):
    """
        Calculates the first error between estimated and true matrix A.

        Parameters:
        A (np.ndarray): The true matrix.
        A_hat (np.ndarray): The estimated matrix.

        Returns:
        float: The calculated error.
    """
    K = A.shape[1]

    all_perm = list(itertools.permutations(range(K)))
    error = float('inf')

    for perm in all_perm:
        error = min(error, np.sum(np.abs(A[:, perm] - A_hat).sum(axis=0)))

    return error


def error2_A(A, A_hat):
    """
        Calculates the second error between estimated and true matrix A.

        Parameters:
        A (np.ndarray): The true matrix.
        A_hat (np.ndarray): The estimated matrix.

        Returns:
        float: The calculated error.
    """
    K = A.shape[1]
    used = np.ones(K)
    A_perm = np.zeros((A.shape[0], A.shape[1]))

    for k in range(K):
        dis = np.sum(np.abs(A - A_hat[:, k][:, np.newaxis]) * used, axis=0)
        index = np.argmin(dis)
        A_perm[:, k] = A[:, index]
        used[index] = float('inf')

    return np.sum(np.abs(A_perm - A_hat).sum(axis=0))


def compute_W_from_AD(A_hat, D):
    """
        Computes matrix W from estimated matrix A and data matrix D.

        Parameters:
        A_hat (np.ndarray): The estimated matrix A.
        D (np.ndarray): The data matrix.

        Returns:
        np.ndarray: The computed matrix W.
    """
    K = A_hat.shape[1]
    n = D.shape[1]

    W_hat = np.zeros((K, n))
    M = np.vstack((np.eye(K-1), -np.ones(K-1)))
    bM = np.eye(K)[:, K-1]
    Dmat = 2 * (A_hat @ M).T @ (A_hat @ M)
    Amat = M.T
    bvec = -bM

    AM = A_hat @ M
    AbM = A_hat @ bM

    for i in range(n):
        dvec = 2 * (D[:, i] - AbM).T @ AM
        x = cp.Variable(len(dvec))
        constraints = [x >= 0, Amat @ x <= bvec]

        objective = cp.Maximize(dvec @ x - cp.quad_form(x, Dmat))

        problem = cp.Problem(objective, constraints)
        problem.solve()

        W_hat[:, i] = np.hstack((x.value, 1 - np.sum(x.value)))

    W_hat = np.maximum(W_hat, 0)

    return W_hat


def replaceWithLeastPositive(vec):
    """
        Replaces non-positive elements in the vector with the least positive element.

        Parameters:
        vec (np.ndarray): The input vector.

        Returns:
        np.ndarray: The updated vector.
    """
    vec[vec <= 0] = np.min(vec[vec > 0])
    return vec


def nearPD(mat):
    """
        Finds the nearest positive definite matrix.

        Parameters:
        mat (np.ndarray): The input matrix.

        Returns:
        np.ndarray: The nearest positive definite matrix.
    """
    mat = (mat + mat.T) / 2
    values, vectors = np.linalg.eig(mat)
    values = replaceWithLeastPositive(values)
    return vectors.dot(np.diag(values)).dot(vectors.T)


def successiveProj(R, K):
    """
        Executes the successive projection algorithm.

        Parameters:
        R (np.ndarray): The input matrix.
        K (int): The number of singular values to compute.

        Returns:
        Tuple: Tuple containing projected R and the indices.
    """
    n = R.shape[0]
    Y = np.hstack((np.ones((n, 1)), R))
    indexSet = []

    while len(indexSet) < K:
        l2Norms = np.sqrt(np.sum(Y ** 2, axis=1))
        index = np.argmax(l2Norms)
        indexSet.append(index)
        u = Y[index, :] / np.sqrt(np.sum(Y[index, :] ** 2))
        Y = Y - np.outer(np.sum(Y * u, axis=1), u)

    return R[indexSet, :], indexSet


def vertices_est_SP(R, m):
    """
       Estimates the vertices using the SP method.

       Parameters:
       R (np.ndarray): The input matrix.
       m (int): The number of vertices.

       Returns:
       Tuple: Tuple containing projected theta and the indices.
    """
    K = R.shape[1] + 1
    kmeans = KMeans(n_clusters=m, max_iter=K*100, n_init=K*10).fit(R)
    theta = kmeans.cluster_centers_
    return successiveProj(theta, K)

