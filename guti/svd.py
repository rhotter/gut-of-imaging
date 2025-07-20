import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from scipy import sparse


def compute_svd_cpu(Jac_cpu):
    # Convert to sparse matrix if it isn't already
    if not sparse.issparse(Jac_cpu):
        Jac_cpu = sparse.csr_matrix(Jac_cpu)

    # For sparse matrices, use scipy's svds
    # k is the number of singular values to compute
    # If k is None, it will compute min(n, m) singular values
    s = svds(Jac_cpu, k=None, return_singular_vectors=False)
    return s


def compute_svd_gpu(Jac_cpu):
    # Move data to GPU and compute SVD
    Jac_gpu = torch.tensor(Jac_cpu, device="cuda")  # Move to GPU
    s = torch.linalg.svdvals(Jac_gpu, driver="gesvd")
    s = s.cpu().numpy()  # Move results back to CPU for plotting
    return s


def plot_svd(s):
    plt.figure()
    plt.semilogy(s / s[0])
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title("Singular value spectrum of J")
    plt.grid(True)
    plt.ylim(1e-5, 1)  # Set y-axis limits from 1e-5 to 1
