import numpy as np

#Some util functions
def calc_eigenspace(L):
    eigenValues, eigenVectors = np.linalg.eigh(L)
    idx = np.argsort(eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    return (eigenValues, eigenVectors)

def calc_laplacian_graph(W):
    W = W - np.diag(np.diag(W))

    D = np.diagflat(np.sum(W, 1))
    L = D - W
    eig_vals, eig_vecs = calc_eigenspace(L)
    return eig_vals, eig_vecs