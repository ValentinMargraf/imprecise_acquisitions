# -------------------------
# Custom 1D test function
# -------------------------
import torch
from botorch.test_functions import SyntheticTestFunction


class OneDimSine(SyntheticTestFunction):
    r"""
    A simple 1D synthetic test function for BoTorch.
    f(x) = sin(3πx) + x² - 0.5
    Domain: x ∈ [0, 1]
    """

    dim = 1
    _bounds = [(0.0, 1.0)]
    _optimal_value = None  # can optionally set to known global min

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        x = X[..., 0]
        return torch.sin(3 * torch.pi * x) + x**2 - 0.5
