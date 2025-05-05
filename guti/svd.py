import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds


def compute_svd_gpu(Jac_cpu):
    # Move data to GPU and compute SVD
    Jac_gpu = torch.tensor(Jac_cpu, device="cuda")  # Move to GPU
    s = torch.linalg.svdvals(Jac_gpu)
    s = s.cpu().numpy()  # Move results back to CPU for plotting
    return s


def compute_svd_batched(Jac_cpu, batch_size=1000):
    """Process matrix in batches to reduce memory requirements"""
    # Compute SVD using batched processing
    device = torch.device("cuda")
    m, n = Jac_cpu.shape

    # Convert to torch tensor if it's numpy
    if isinstance(Jac_cpu, np.ndarray):
        Jac = torch.from_numpy(Jac_cpu)
    else:
        Jac = Jac_cpu

    batches = []
    for i in range(0, m, batch_size):
        end = min(i + batch_size, m)
        batch = Jac[i:end].to(device)
        u, s, vh = torch.linalg.svd(batch, full_matrices=False)
        # We only need singular values for now
        batches.append(s.cpu().numpy())
        torch.cuda.empty_cache()  # Free GPU memory

    # Combine results (simplistic - just concatenate the singular values)
    # For a proper implementation, you'd need to combine the U, S, V matrices
    return np.concatenate(batches)


def plot_svd(s):
    plt.figure()
    plt.semilogy(s)
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title("Singular value spectrum of J")
    plt.grid(True)
