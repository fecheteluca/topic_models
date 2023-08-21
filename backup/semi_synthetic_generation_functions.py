import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from scipy.special import softmax


def semi_synthetic_generation(dir, dataset, K, N=None, seed=None):
    data = pd.read_csv(f"{dir}/real_data/{dataset}.txt", header=None, sep="")
    data = data.values

    if N is None:
        out_txt = f"semi_synthetic_{K}_{dataset}.txt"
        out_A = f"semi_synthetic_{K}_{dataset}.A"
    else:
        out_txt = f"semi_synthetic_{K}_{N}_{dataset}.txt"
        out_A = f"semi_synthetic_{K}_{N}_{dataset}.A"

    n = np.max(data[:, 0])
    p = np.max(data[:, 1])
    D = np.zeros((n, p))
    for i in range(data.shape[0]):
        D[data[i, 0], data[i, 1]] = data[i, 2]

    if N is None:
        N = np.sum(D, axis=1)
    elif isinstance(N, int):
        N = np.repeat(N, n)

    if seed is None:
        LDA_obj = LatentDirichletAllocation(n_components=K, learning_method='batch')
    else:
        LDA_obj = LatentDirichletAllocation(n_components=K, learning_method='batch', random_state=seed)

    D_sim = LDA_obj.fit_transform(D) @ softmax(LDA_obj.components_, axis=1).T
    D_synth = np.zeros((n, p))
    for i in range(len(N)):
        D_synth[i, :] = np.random.multinomial(N[i], D_sim[i, :])

    with open(f"{dir}/semi_synthetic_data/{out_txt}", 'w') as fileConn:
        for i in range(n):
            for j in range(p):
                if D_synth[i, j] > 0:
                    fileConn.write(f"{i} {j} {D_synth[i, j]}\n")

    np.savetxt(out_A, np.exp(LDA_obj.components_).T)
