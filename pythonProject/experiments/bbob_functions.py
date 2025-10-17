import torch
from torch import Tensor
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.acquisition import AcquisitionFunction, UpperConfidenceBound, ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

device = torch.device("cpu")
dtype = torch.float64
# -------------------------
# Utility: rescale [0,1] → [a,b]
# -------------------------
def rescale(X: Tensor, bounds: torch.Tensor) -> Tensor:
    """Rescale from [0,1]^d to [low, high]^d given bounds tensor of shape (2, d)."""
    low, high = bounds[0], bounds[1]
    return low + (high - low) * X

# -------------------------
# Rosenbrock 4D
# -------------------------
def rosenbrock4(X: Tensor, noise_std: float = 0.0) -> Tensor:
    X_ = X.unsqueeze(0) if X.ndim == 1 else X
    # Rosenbrock: sum_{i=1}^{d-1} [100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2]
    f = torch.sum(100.0 * (X_[:, 1:] - X_[:, :-1]**2)**2 + (1 - X_[:, :-1])**2, dim=-1)
    if noise_std > 0:
        f = f + noise_std * torch.randn_like(f)
    return f


# -------------------------
# Hartmann 6D
# -------------------------
def hartmann6(X: Tensor, noise_std: float = 0.0) -> Tensor:
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=X.dtype, device=X.device)
    A = torch.tensor([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14]
    ], dtype=X.dtype, device=X.device)
    P = 1e-4 * torch.tensor([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381]
    ], dtype=X.dtype, device=X.device)

    X_ = X.unsqueeze(0) if X.ndim == 1 else X
    total = torch.zeros(X_.shape[0], dtype=X.dtype, device=X.device)
    for i in range(4):
        total += alpha[i] * torch.exp(-torch.sum(A[i] * (X_ - P[i])**2, dim=-1))
    f = -total  # minimization
    if noise_std > 0:
        f = f + noise_std * torch.randn_like(f)
    return f


# -------------------------
# Branin 2D
# -------------------------
def branin2(X: Tensor, noise_std: float = 0.0) -> Tensor:
    # X is 2D
    a = 1.0
    b = 5.1 / (4.0 * torch.pi**2)
    c = 5.0 / torch.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * torch.pi)

    X_ = X.unsqueeze(0) if X.ndim == 1 else X
    x1 = X_[:, 0]
    x2 = X_[:, 1]
    f = a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * torch.cos(x1) + s
    if noise_std > 0:
        f = f + noise_std * torch.randn_like(f)
    return f
# -------------------------
# Hartmann3 function
# -------------------------
def hartmann3(X: Tensor, noise_std: float = 0.0) -> Tensor:
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=dtype, device=device)
    A = torch.tensor([[3.0, 10.0, 30.0],
                      [0.1, 10.0, 35.0],
                      [3.0, 10.0, 30.0],
                      [0.1, 10.0, 35.0]], dtype=dtype, device=device)
    P = torch.tensor([[3689, 1170, 2673],
                      [4699, 4387, 7470],
                      [1091, 8732, 5547],
                      [381, 5743, 8828]], dtype=dtype, device=device) * 1e-4
    X_ = X.unsqueeze(0) if X.ndim == 1 else X
    total = torch.zeros(X_.shape[0], dtype=dtype, device=device)
    for i in range(4):
        total += alpha[i] * torch.exp(-torch.sum(A[i] * (X_ - P[i])**2, dim=-1))
    y = -total  # minimization problem
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)

    return y

# -------------------------
# Ackley 6D
# -------------------------
def ackley6(X: Tensor, noise_std: float = 0.0) -> Tensor:
    X_ = X.unsqueeze(0) if X.ndim == 1 else X
    d = 6
    assert X_.shape[1] == d, f"Expected input dimension {d}, got {X_.shape[1]}"
    term1 = -20.0 * torch.exp(-0.2 * torch.sqrt(torch.mean(X_**2, dim=1)))
    term2 = -torch.exp(torch.mean(torch.cos(2 * torch.pi * X_), dim=1))
    f = term1 + term2 + 20 + torch.e
    if noise_std > 0:
        f = f + noise_std * torch.randn_like(f)
    return f

# -------------------------
# Rastrigin 8D
# -------------------------
def rastrigin8(X: Tensor, noise_std: float = 0.0) -> Tensor:
    X_ = X.unsqueeze(0) if X.ndim == 1 else X
    d = 8
    assert X_.shape[1] == d, f"Expected input dimension {d}, got {X_.shape[1]}"
    f = torch.sum(X_**2 - 10 * torch.cos(2 * torch.pi * X_) + 10, dim=-1)
    if noise_std > 0:
        f = f + noise_std * torch.randn_like(f)
    return f

# -------------------------
# Griewank 10D
# -------------------------
def griewank10(X: Tensor, noise_std: float = 0.0) -> Tensor:
    X_ = X.unsqueeze(0) if X.ndim == 1 else X
    d = 10
    assert X_.shape[1] == d, f"Expected input dimension {d}, got {X_.shape[1]}"
    term1 = torch.sum(X_**2 / 4000.0, dim=-1)
    term2 = torch.prod(torch.cos(X_ / torch.sqrt(torch.arange(1, d+1, dtype=X.dtype, device=X.device))), dim=-1)
    f = term1 - term2 + 1
    if noise_std > 0:
        f = f + noise_std * torch.randn_like(f)
    return f

# -------------------------
# Griewank 3D
# -------------------------
def griewank3(X: Tensor, noise_std: float = 0.0) -> Tensor:
    X_ = X.unsqueeze(0) if X.ndim == 1 else X
    d = 3
    assert X_.shape[1] == d, f"Expected input dimension {d}, got {X_.shape[1]}"
    term1 = torch.sum(X_**2 / 4000.0, dim=-1)
    term2 = torch.prod(torch.cos(X_ / torch.sqrt(torch.arange(1, d+1, dtype=X.dtype, device=X.device))), dim=-1)
    f = term1 - term2 + 1
    if noise_std > 0:
        f = f + noise_std * torch.randn_like(f)
    return f

# -------------------------
# Rastrigin 4D
# -------------------------
def rastrigin4(X: Tensor, noise_std: float = 0.0) -> Tensor:
    X_ = X.unsqueeze(0) if X.ndim == 1 else X
    d = 4
    assert X_.shape[1] == d, f"Expected input dimension {d}, got {X_.shape[1]}"
    f = torch.sum(X_**2 - 10 * torch.cos(2 * torch.pi * X_) + 10, dim=-1)
    if noise_std > 0:
        f = f + noise_std * torch.randn_like(f)
    return f
