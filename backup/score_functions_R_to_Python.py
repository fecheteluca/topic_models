from itertools import permutations, combinations
import numpy as np
from scipy.sparse.linalg import svds
from scipy.optimize import nnls
from scipy.optimize import minimize
from sklearn.cluster import KMeans


def score(K, K0, m, D, scatterplot=False):
    p = D.shape[0]
    obj = svds(D, k=K)
    Xi = obj[0]

    Xi[:, 0] = np.abs(Xi[:, 0])
    R = np.apply_along_axis(lambda x: x / Xi[:, 0], axis=0, arr=Xi[:, 1:K])

    vertices_est_obj = vertices_est(R, K0, m)
    V = vertices_est_obj['V']
    theta = vertices_est_obj['theta']

    if scatterplot:
        import matplotlib.pyplot as plt
        plt.plot(R[:, 0], R[:, 1])
        plt.scatter(V[:, 0], V[:, 1], c='red', s=50)
        plt.show()

    Pi = np.linalg.solve(np.hstack((V, np.ones((K, 1)))), np.hstack((R, np.ones((p, 1)))))
    Pi = np.maximum(Pi, 0)
    temp = np.sum(Pi, axis=1)
    Pi = np.apply_along_axis(lambda x: x / temp, axis=1, arr=Pi)

    A_hat = Xi[:, 0] * Pi

    temp = np.sum(A_hat, axis=0)
    A_hat = np.apply_along_axis(lambda x: x / temp, axis=1, arr=A_hat)

    return {'A_hat': A_hat, 'R': R, 'V': V, 'Pi': Pi, 'theta': theta}


def norm_score_N(K, K0, m, N, D, scatterplot=False):
    p = D.shape[0]
    n = D.shape[1]
    M = np.sum(D / np.tile(N, (n, 1)).T, axis=1)
    obj = svds(np.sqrt(np.diag(M ** (-1))) @ D, k=K)
    Xi = obj[0]

    Xi[:, 0] = np.abs(Xi[:, 0])
    R = np.apply_along_axis(lambda x: x / Xi[:, 0], axis=0, arr=Xi[:, 1:K])

    vertices_est_obj = vertices_est(R, K0, m)
    V = vertices_est_obj['V']
    theta = vertices_est_obj['theta']

    if scatterplot:
        import matplotlib.pyplot as plt
        plt.plot(R[:, 0], R[:, 1])
        plt.scatter(V[:, 0], V[:, 1], c='red', s=50)
        plt.show()

    Pi = np.linalg.solve(np.hstack((V, np.ones((K, 1)))), np.hstack((R, np.ones((p, 1)))))
    Pi = np.maximum(Pi, 0)
    temp = np.sum(Pi, axis=1)
    Pi = np.apply_along_axis(lambda x: x / temp, axis=1, arr=Pi)

    A_hat = np.sqrt(np.diag(M)) @ Xi[:, 0] @ Pi

    temp = np.sum(A_hat, axis=0)
    A_hat = np.apply_along_axis(lambda x: x / temp, axis=1, arr=A_hat)

    return {'A_hat': A_hat, 'R': R, 'V': V, 'Pi': Pi, 'theta': theta}


def norm_score(K, K0, m, D, Mquantile=0, scatterplot=False, VHMethod='SVS'):
    p = D.shape[0]
    n = D.shape[1]
    M = np.mean(D, axis=1)
    M_trunk = np.minimum(M, np.quantile(M, Mquantile))
    obj = svds(np.sqrt(np.diag(M_trunk ** (-1))) @ D, k=K)
    Xi = obj[0]

    Xi[:, 0] = np.abs(Xi[:, 0])
    R = np.apply_along_axis(lambda x: x / Xi[:, 0], axis=0, arr=Xi[:, 1:K])

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
        import matplotlib.pyplot as plt
        plt.plot(R[:, 0], R[:, 1])
        plt.scatter(V[:, 0], V[:, 1], c='red', s=50)
        plt.show()

    Pi = np.linalg.solve(np.hstack((V, np.ones((K, 1)))), np.hstack((R, np.ones((p, 1)))))
    Pi = np.maximum(Pi, 0)
    temp = np.sum(Pi, axis=1)
    Pi = np.apply_along_axis(lambda x: x / temp, axis=1, arr=Pi)

    A_hat = np.sqrt(np.diag(M_trunk)) @ Xi[:, 0] @ Pi

    temp = np.sum(A_hat, axis=0)
    A_hat = np.apply_along_axis(lambda x: x / temp, axis=1, arr=A_hat)

    return {'A_hat': A_hat, 'R': R, 'V': V, 'Pi': Pi, 'theta': theta}


def score_W(K, K0, m, D):
    p = D.shape[0]
    obj = svds(D, k=K)
    Xi = obj[2].T

    Xi[:, 0] = np.abs(Xi[:, 0])
    R = np.apply_along_axis(lambda x: x / Xi[:, 0], axis=0, arr=Xi[:, 1:K])

    vertices_est_obj = vertices_est(R, K0, m)
    V = vertices_est_obj['V']

    Pi = np.linalg.solve(np.hstack((V, np.ones((K, 1)))), np.hstack((R, np.ones((p, 1)))))
    Pi = np.maximum(Pi, 0)
    temp = np.sum(Pi, axis=1)
    Pi = np.apply_along_axis(lambda x: x / temp, axis=1, arr=Pi)

    W_tilde = Pi * Xi[:, 0]
    W_tilde_norm = np.apply_along_axis(lambda x: x - np.mean(x), axis=1, arr=W_tilde)
    Q_direction = np.linalg.eig(np.dot(W_tilde_norm.T, W_tilde_norm))[1][:, K - 1]

    W_hat = np.apply_along_axis(lambda x: x / np.sum(x), axis=1, arr=W_tilde * np.abs(Q_direction))

    A_hat = np.zeros((p, K))
    for i in range(p):
        A_hat[i, :] = nnls(W_hat.T, D[i, :])[0]
    for k in range(K):
        A_hat[:, k] = A_hat[:, k] / np.sum(A_hat[:, k])

    return {'A_hat': A_hat, 'W_hat': W_hat, 'R': R, 'V': V, 'Pi': Pi}


def debias_score_N(K, K0, m, N, D, scatterplot=False):
    p = D.shape[0]
    n = D.shape[1]
    M = np.sum(D / np.tile(N, (n, 1)).T, axis=1)
    obj = svds(D @ D.T - np.diag(M), k=K)
    Xi = obj[0]

    Xi[:, 0] = np.abs(Xi[:, 0])
    R = np.apply_along_axis(lambda x: x / Xi[:, 0], axis=0, arr=Xi[:, 1:K])

    vertices_est_obj = vertices_est(R, K0, m)
    V = vertices_est_obj['V']
    theta = vertices_est_obj['theta']

    if scatterplot:
        import matplotlib.pyplot as plt
        plt.plot(R[:, 0], R[:, 1])
        plt.scatter(V[:, 0], V[:, 1], c='red', s=50)
        plt.show()

    Pi = np.linalg.solve(np.hstack((V, np.ones((K, 1)))), np.hstack((R, np.ones((p, 1)))))
    Pi = np.maximum(Pi, 0)
    temp = np.sum(Pi, axis=1)
    Pi = np.apply_along_axis(lambda x: x / temp, axis=1, arr=Pi)

    A_hat = Xi[:, 0] * Pi

    temp = np.sum(A_hat, axis=0)
    A_hat = np.apply_along_axis(lambda x: x / temp, axis=1, arr=A_hat)

    return {'A_hat': A_hat, 'R': R, 'V': V, 'Pi': Pi, 'theta': theta}


def debias_score(K, K0, m, N, D, scatterplot=False):
    p = D.shape[0]
    n = D.shape[1]
    M = np.mean(D, axis=1)
    obj = svds(D @ D.T - n / np.mean(N) * np.diag(M), k=K)
    Xi = obj[0]

    Xi[:, 0] = np.abs(Xi[:, 0])
    R = np.apply_along_axis(lambda x: x / Xi[:, 0], axis=0, arr=Xi[:, 1:K])

    vertices_est_obj = vertices_est(R, K0, m)
    V = vertices_est_obj['V']
    theta = vertices_est_obj['theta']

    if scatterplot:
        import matplotlib.pyplot as plt
        plt.plot(R[:, 0], R[:, 1])
        plt.scatter(V[:, 0], V[:, 1], c='red', s=50)
        plt.show()

    Pi = np.linalg.solve(np.hstack((V, np.ones((K, 1)))), np.hstack((R, np.ones((p, 1)))))
    Pi = np.maximum(Pi, 0)
    temp = np.sum(Pi, axis=1)
    Pi = np.apply_along_axis(lambda x: x / temp, axis=1, arr=Pi)

    A_hat = Xi[:, 0] * Pi

    temp = np.sum(A_hat, axis=0)
    A_hat = np.apply_along_axis(lambda x: x / temp, axis=1, arr=A_hat)

    return {'A_hat': A_hat, 'R': R, 'V': V, 'Pi': Pi, 'theta': theta}


def vertices_est(R, K0, m):
    K = R.shape[1] + 1

    obj = KMeans(n_clusters=m, max_iter=K * 100, n_init=K * 10)
    obj.fit(R)
    theta = obj.cluster_centers_
    theta_original = theta

    if K0 > 2:
        for k0 in range(3, K0 + 1):
            inner = np.dot(theta, theta.T)
            distance = np.diag(inner)[:, None] @ np.ones((1, len(np.diag(inner)))) + np.ones((len(np.diag(inner))),
                                                                                             1) @ np.diag(inner)[None,
                                                                                                  :] - 2 * inner
            ave_dist = np.mean(distance, axis=0)
            index = np.argmax(ave_dist)
            theta0 = np.vstack((theta0, theta[index, :]))
            theta = np.delete(theta, index, axis=0)
        theta = theta0

    comb = np.array(list(combinations(range(K0), K)))
    max_values = np.zeros(comb.shape[1])
    for i in range(comb.shape[1]):
        for j in range(K0):
            max_values[i] = max(simplex_dist(theta[j, :], theta[comb[:, i], :]), max_values[i])
    min_index = np.argmin(max_values)

    return {'V': theta[comb[:, min_index], :], 'theta': theta_original}


def simplex_dist(theta, V):
    VV = np.hstack((np.diag(np.ones(V.shape[1] - 1)), -np.ones((V.shape[1] - 1, 1)))) @ V
    D = VV @ VV.T
    d = VV @ (theta - V[-1, :])
    A = np.hstack((np.diag(np.ones(V.shape[1] - 1)), -np.ones((V.shape[1] - 1, 1))))
    b0 = np.hstack((np.zeros(V.shape[1] - 1), -1))

    res = minimize(lambda x: np.sum((theta - V[-1, :]) ** 2) + 2 * np.dot(x, np.dot(D, x)), np.zeros(V.shape[1] - 1),
                   constraints={'type': 'eq', 'fun': lambda x: np.dot(A, x) - b0})

    return np.sum((theta - V[-1, :]) ** 2) + 2 * res.fun


def error1_A(A, A_hat):
    K = A.shape[1]
    all_perm = list(permutations(range(K)))
    error = np.inf
    for i in range(len(all_perm)):
        error = min(error, np.sum(np.sum(np.abs(A[:, list(all_perm[i])] - A_hat), axis=0)))
    return error


def error2_A(A, A_hat):
    K = A.shape[1]
    used = np.ones(K)
    A_perm = np.zeros(A.shape)
    for k in range(K):
        dis = np.sum(np.abs(A - A_hat[:, k]), axis=0) * used
        index = np.argmin(dis)
        A_perm[:, k] = A[:, index]
        used[index] = np.inf
    return np.sum(np.sum(np.abs(A_perm - A_hat), axis=0))


def compute_W_from_AD(A_hat, D):
    K = A_hat.shape[1]
    n = D.shape[1]
    W_hat = np.zeros((K, n))
    M = np.vstack((np.diag(K - 1), -np.ones((K - 1, K - 1)))) @ A_hat.T
    bM = np.diag(K)[:, K - 1]
    Dmat = 2 * M.T @ M
    Amat = M.T
    bvec = -bM
    AM = A_hat @ M
    AbM = A_hat @ bM
    for i in range(n):
        dvec = 2 * (D[:, i] - AbM) @ AM

        qp_sol = minimize(lambda x: np.dot(x, np.dot(Dmat, x)) + np.dot(dvec, x), np.zeros(K),
                          constraints={'type': 'eq', 'fun': lambda x: np.dot(Amat, x) - bvec}).x
        W_hat[:, i] = np.hstack((qp_sol, 1 - np.sum(qp_sol)))
    W_hat = np.maximum(W_hat, 0)
    return W_hat


def replaceWithLeastPositive(vec):
    vec[vec <= 0] = np.min(vec[vec > 0])
    return vec


def nearPD(mat):
    mat = (mat + mat.T) / 2
    eigenObj = np.linalg.eig(mat)
    values = eigenObj[0]
    vectors = eigenObj[1]
    values = replaceWithLeastPositive(values)
    return vectors @ np.diag(values) @ vectors.T


def successiveProj(R, K):
    n = R.shape[0]
    Y = np.hstack((np.ones((n, 1)), R))
    indexSet = []
    while len(indexSet) < K:
        l2Norms = np.sqrt(np.sum(Y ** 2, axis=1))
        index = np.argmax(l2Norms)
        indexSet.append(index)
        u = Y[index, :] / np.sqrt(np.sum(Y[index, :] ** 2))
        Y = Y - np.outer(np.sum(Y * u, axis=1), u)
    return {'V': R[indexSet, :], 'indexSet': indexSet}


def vertices_est_SP(R, m):
    K = R.shape[1] + 1
    obj = KMeans(n_clusters=m, max_iter=K * 100, n_init=K * 10)
    obj.fit(R)
    theta = obj.cluster_centers_
    return successiveProj(theta, K)
