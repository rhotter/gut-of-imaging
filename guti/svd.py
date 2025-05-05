import torch
import numpy as np
import matplotlib.pyplot as plt


def compute_svd_gpu(Jac_cpu):
    # Move data to GPU and compute SVD
    Jac_gpu = torch.tensor(Jac_cpu, device="cuda")  # Move to GPU
    s = torch.linalg.svdvals(Jac_gpu)
    s = s.cpu().numpy()  # Move results back to CPU for plotting
    return s


def plot_svd(s):
    plt.figure()
    plt.semilogy(s)
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title("Singular value spectrum of J")
    plt.grid(True)
