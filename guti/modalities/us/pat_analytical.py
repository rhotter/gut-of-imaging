import torch
import numpy as np
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────────────────────
R     = 1.0
r     = 0.5
alpha = 0.3
h     = 0.2
beta  = 2.0

# Integration window half-width and grid size for Δφ, Δγ
D     = 1.0
n_int = 50

# Sampling grid for center angles φ, γ
n_plot     = 50
phi_vals   = torch.linspace(0, 2*np.pi, n_plot)
gamma_vals = torch.linspace(0, 2*np.pi, n_plot)

# Precompute Δφ, Δγ grids
dp = torch.linspace(-D, D, n_int)
dg = torch.linspace(-D, D, n_int)
DP, DG = torch.meshgrid(dp, dg, indexing='xy')

# ── Integrand function ───────────────────────────────────────────────────────
def integrand(Dphi, Dgamma, gamma0, phi0):
    t = torch.tan(alpha)
    d = torch.sqrt(
        (r * torch.sin(gamma0) - R*t)**2
      + R**2
      + (r * torch.cos(gamma0))**2
      - 2*r*R*torch.cos(phi0)
    )
    term1 = h**2 + (r*Dphi)**2 + (r*Dgamma)**2
    term2 = (
        d
      + Dgamma * r*R*t*torch.cos(gamma0) / d
      - r*R*torch.sin(phi0)   * Dphi        / d
    )
    expo = (
        -beta * term1 * term2 / d**2
      + alpha*( d 
               - r*R*t*torch.cos(gamma0)/d
               + r*R*torch.sin(phi0)/d )
    )
    return torch.exp(expo)

# ── Inner 2D trapezoidal integrator ───────────────────────────────────────────
def inner_integral(gamma0, phi0):
    vals = integrand(DP, DG, gamma0, phi0)
    I_phi   = torch.trapz(vals, dp, dim=0)
    I_total = torch.trapz(I_phi, dg, dim=0)
    return I_total

# ── Build 2D field and plot ───────────────────────────────────────────────────
F = torch.zeros(n_plot, n_plot)
for i, g in enumerate(gamma_vals):
    for j, p in enumerate(phi_vals):
        F[i, j] = inner_integral(g, p)

# Convert to NumPy for plotting
F_np = F.cpu().numpy()

plt.figure(figsize=(6,5))
plt.imshow(
    F_np,
    origin='lower',
    extent=(0, 2*np.pi, 0, 2*np.pi),
    aspect='auto'
)
plt.xlabel(r'$\gamma$')
plt.ylabel(r'$\phi$')
plt.title(r'$\displaystyle\int e^{\dots}\,d\phi\,d\gamma$ vs $\gamma,\phi$')
plt.colorbar(label='Integral value')
plt.tight_layout()
plt.show()
