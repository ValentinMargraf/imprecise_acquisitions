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
from gaussian_process_models import pyro_gp_model, run_mcmc, build_gp_from_posterior, mixture_predict
from torch.distributions import Normal


device = torch.device("cpu")
dtype = torch.float64

# Mixture UCB: use hierarchical predictive mean & variance
class MixtureUCB(AcquisitionFunction):
    def __init__(self, gp_models, beta: float = 1.0, maximize: bool = False):
        super().__init__(gp_models[0])
        self.gp_models = gp_models
        self.beta = beta
        self.maximize = maximize

    def forward(self, X):
        if X.dim() == 1:
            X = X.unsqueeze(0)

        # mixture_predict should return per-model mus and variances (and optionally sigmas)
        mus, vars_, sigmas = mixture_predict(self.gp_models, X)
        # normalize shapes: expect (M, batch, 1) or (M, batch)
        if mus.size(-1) == 1:
            mus = mus.squeeze(-1)    # -> (M, batch)
        if vars_ is not None and vars_.size(-1) == 1:
            vars_ = vars_.squeeze(-1)

        # mixture mean
        mu_mix = mus.mean(dim=0)  # shape (batch,)

        # mixture variance: E[mu^2 + sigma^2] - mu_mix^2
        second_moment = (mus.pow(2) + vars_).mean(dim=0)
        var_mix = second_moment - mu_mix.pow(2)
        var_mix = var_mix.clamp_min(1e-12)   # numeric safety
        std_mix = var_mix.sqrt()
        if self.maximize:
            ucb_mix = mu_mix + (self.beta ** 0.5) * std_mix
        else:
            ucb_mix = - mu_mix + (self.beta ** 0.5) * std_mix

        return ucb_mix.squeeze()



class MixtureEI(AcquisitionFunction):
    def __init__(self, gp_models, best_f: float, maximize: bool = True):
        super().__init__(gp_models[0])
        self.gp_models = gp_models
        self.best_f = best_f
        self.maximize = maximize

    def forward(self, X):
        if X.dim() == 1:
            X = X.unsqueeze(0)

        # --- Step 1: per-model predictions ---
        mus, vars_, _ = mixture_predict(self.gp_models, X)
        if mus.size(-1) == 1:
            mus = mus.squeeze(-1)
        if vars_ is not None and vars_.size(-1) == 1:
            vars_ = vars_.squeeze(-1)

        # --- Step 2: mixture mean & variance ---
        mu_mix = mus.mean(dim=0)
        second_moment = (mus.pow(2) + vars_).mean(dim=0)
        var_mix = (second_moment - mu_mix.pow(2)).clamp_min(1e-12)
        std_mix = var_mix.sqrt()

        # --- Step 3: Expected Improvement ---
        normal = Normal(torch.zeros_like(mu_mix), torch.ones_like(mu_mix))

        if self.maximize:
            u = (mu_mix - self.best_f) / std_mix
        else:
            u = (self.best_f - mu_mix) / std_mix

        cdf = normal.cdf(u)
        pdf = normal.log_prob(u).exp()

        if self.maximize:
            ei = (mu_mix - self.best_f) * cdf + std_mix * pdf
        else:
            ei = (self.best_f - mu_mix) * cdf + std_mix * pdf

        # For numerical stability: EI=0 where std_mix≈0
        ei = torch.where(std_mix < 1e-9, torch.zeros_like(ei), ei)

        return ei.squeeze()


def sd_acqf_optimize_median(gp_models, bounds, maximize=False, beta=1.0, num_restarts=20, raw_samples=512, acqfunc="UCB",best_f=None):
    r"""
    Stochastic Dominance-based acquisition optimizer (median variant).

    Steps:
    1. For each GP model, optimize its UCB acquisition function.
    2. Collect all candidate points.
    3. Evaluate plausible q-values across GP models at each candidate.
    4. Filter dominated points (keep only non-dominated).
    5. Among non-dominated, choose the median candidate (based on plausibility score).

    Args:
        gp_models: list of GP models (posterior samples).
        bounds: tensor of shape (2, d) with box constraints.
        beta: UCB parameter.
        num_restarts: restarts for each candidate optimization.
        raw_samples: raw samples for acquisition optimization.

    Returns:
        X_next: tensor of shape (1, d), the selected candidate.
    """
    candidates = []
    for gp in gp_models:
        acq_j = UpperConfidenceBound(gp, beta=beta, maximize=maximize) if acqfunc=="UCB" \
            else ExpectedImprovement(gp, best_f=best_f, maximize=False)
        x_j, _ = optimize_acqf(
            acq_j, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples
        )
        candidates.append(x_j)
    candidates = torch.cat(candidates, dim=0)

    # Compute coordinate-wise median
    x_median = candidates.median(dim=0).values  # shape: (d,)

    # Compute Euclidean distances to the median
    dists = torch.norm(candidates - x_median, dim=1)

    # Select the candidate closest to the median
    median_idx = torch.argmin(dists)
    return candidates[median_idx].unsqueeze(0)

def sd_acqf_optimize(gp_models, bounds, maximize=False, beta=1.0, num_restarts=20, raw_samples=512, select_lowest_variance=True, acqfunc="UCB",best_f=None):
    r"""
    Stochastic Dominance-based acquisition optimizer.

    Steps:
    1. For each GP model, optimize its UCB acquisition function.
    2. Collect all candidate points.
    3. Evaluate plausible q-values across GP models at each candidate.
    4. Filter dominated points (keep only non-dominated).
    5. Among non-dominated, choose the one with smallest variance if select_lowest_variance=True.

    Args:
        gp_models: list of GP models (posterior samples).
        bounds: tensor of shape (2, d) with box constraints.
        beta: UCB parameter.
        num_restarts: restarts for each candidate optimization.
        raw_samples: raw samples for acquisition optimization.

    Returns:
        X_next: tensor of shape (1, d), the selected candidate.
    """
    candidates = []
    for gp in gp_models:
        acq_j = UpperConfidenceBound(gp, beta=beta, maximize=maximize) if acqfunc == "UCB" \
            else ExpectedImprovement(gp, best_f=best_f, maximize=False)
        x_j, _ = optimize_acqf(
            acq_j, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples
        )
        candidates.append(x_j)
    candidates = torch.cat(candidates, dim=0)

    plausibles, variances, mus_ = [], [], []
    for cand in candidates:
        mus, vars_, sigmas = mixture_predict(gp_models, cand.unsqueeze(0))
        q_vals = mus + sigmas
        plausibles.append(q_vals.squeeze())
        mus_.append(mus.mean().item())
        variances.append(q_vals.var().item())

    dominated = set()

    for i in range(len(plausibles)):
        for j in range(len(plausibles)):
            if i == j:
                continue
            # this always tests "one direction", since we always want to maximize the AF
            dom_type = check_dominance(plausibles[i], plausibles[j])
            if dom_type > 0:
                dominated.add(i)

    non_dominated = [k for k in range(len(candidates)) if k not in dominated]
    chosen_idx = min(non_dominated, key=lambda k: variances[k]) if select_lowest_variance else min(non_dominated, key=lambda k: mus_[k])

    return candidates[chosen_idx].unsqueeze(0)

class RiskAwareUCB(AcquisitionFunction):
    r"""
        Risk-aware Upper Confidence Bound (UCB).

        For a set of GP models (e.g., posterior samples),
        compute plausible UCB values:
            q_j(x) = mu_j(x) + sqrt(beta) * sigma_j(x).

        Apply the risk functional:
            rho(Q(x)) = E[q] + λ Var[q].

        Args:
            gp_models: list of GP models (e.g., posterior samples).
            beta: exploration parameter in UCB (default: 1.0).
            lam: risk parameter λ (0 = risk-neutral,
                 >0 = risk-affine, <0 = risk-averse).
            maximize: whether the BO problem is a maximization.
        """
    def __init__(self, gp_models, beta=1.0, lam=0.0, maximize=False, select_ucb = False, select_min=False, select_max=False, select_median=False):
        super().__init__(gp_models[0])
        self.gp_models = gp_models
        self.beta = beta
        self.lam = lam  # lambda parameter for risk preference
        self.maximize = maximize
        self.select_ucb = select_ucb
        self.select_min = select_min
        self.select_max = select_max
        self.select_median = select_median

    def forward(self, X):
        if X.ndim == 1:
            X = X.unsqueeze(0)
        # get plausible acquisition values across GP hyperparameter samples
        mus, vars_, sigmas = mixture_predict(self.gp_models, X)

        if self.maximize:
            q_values = mus + self.beta**0.5 * sigmas   # shape: (num_models, batch)
        else:
            q_values = - mus + self.beta**0.5 * sigmas   # shape: (num_models, batch)


        # mean and variance across plausible values
        mean_q = q_values.mean(dim=0)
        var_q = q_values.var(dim=0)
        min_q = q_values.min(dim=0).values
        max_q = q_values.max(dim=0).values
        median_q = q_values.median(dim=0).values

        # risk functional: E[q] + λ Var[q]

        if self.select_ucb:
            val = mean_q + self.lam * var_q
        elif self.select_min:
            val = min_q
        elif self.select_max:
            val = max_q
        elif self.select_median:
            val = median_q

        return val.squeeze()


class RiskAwareEI(AcquisitionFunction):
    r"""
    Risk-aware Expected Improvement (EI).

    For a set of GP models (e.g., posterior samples),
    compute plausible EI values for each GP:
        q_j(x) = EI_j(x)

    Apply a risk functional across the set of EI values:
        rho(Q(x)) = E[q] + λ Var[q]

    Args:
        gp_models: list of GP models (e.g., posterior samples)
        best_f: reference value for EI
        lam: risk parameter λ (0 = risk-neutral,
             >0 = risk-affine, <0 = risk-averse)
        maximize: whether the BO problem is a maximization
        select_ei: use mean+λ*var
        select_min: select min EI across models
        select_max: select max EI across models
        select_median: select median EI across models
    """
    def __init__(
        self,
        gp_models,
        best_f,
        lam=0.0,
        maximize=False,
        select_ucb=True,
        select_min=False,
        select_max=False,
        select_median=False,
    ):
        super().__init__(gp_models[0])
        self.gp_models = gp_models
        self.best_f = best_f
        self.lam = lam
        self.maximize = maximize
        self.select_ucb = select_ucb
        self.select_min = select_min
        self.select_max = select_max
        self.select_median = select_median

    def forward(self, X):
        if X.ndim == 1:
            X = X.unsqueeze(0)

        # compute EI for each GP
        q_values = torch.stack([
            ExpectedImprovement(gp, best_f=self.best_f, maximize=self.maximize)(X)
            for gp in self.gp_models
        ], dim=0)  # shape: (num_models, batch)

        # risk aggregation
        mean_q = q_values.mean(dim=0)
        var_q = q_values.var(dim=0)
        min_q = q_values.min(dim=0).values
        max_q = q_values.max(dim=0).values
        median_q = q_values.median(dim=0).values

        if self.select_ucb:
            val = mean_q + self.lam * var_q
        elif self.select_min:
            val = min_q
        elif self.select_max:
            val = max_q
        elif self.select_median:
            val = median_q
        else:
            raise ValueError("No valid risk aggregation selected.")

        return val.squeeze()

def check_fsd(a: torch.Tensor, b: torch.Tensor, n_grid=200):
    """
    Check if `a` is first-order stochastically dominated by `b`.
    """
    z_min = min(a.min(), b.min()).item()
    z_max = max(a.max(), b.max()).item()
    zs = torch.linspace(z_min, z_max, n_grid)

    Fa = (a.view(-1, 1) <= zs).float().mean(dim=0)
    Fb = (b.view(-1, 1) <= zs).float().mean(dim=0)

    return torch.all(Fa >= Fb) and torch.any(Fa > Fb)


def check_ssd(a: torch.Tensor, b: torch.Tensor, n_grid=200):
    """
    Check if `a` is second-order stochastically dominated by `b`.
    Only called if FSD fails.
    """
    z_min = min(a.min(), b.min()).item()
    z_max = max(a.max(), b.max()).item()
    zs = torch.linspace(z_min, z_max, n_grid)
    dz = zs[1] - zs[0]

    Fa = (a.view(-1, 1) <= zs).float().mean(dim=0)
    Fb = (b.view(-1, 1) <= zs).float().mean(dim=0)

    Ia = torch.cumsum(Fa, dim=0) * dz
    Ib = torch.cumsum(Fb, dim=0) * dz

    return torch.all(Ia >= Ib) and torch.any(Ia > Ib)


def check_dominance(a: torch.Tensor, b: torch.Tensor, n_grid=200):
    """
    Returns:
        1  if a is FSD-dominated by b
        2  if a is SSD-dominated by b (and not FSD)
        0  if no dominance
    """
    if check_fsd(a, b, n_grid=n_grid):
        return 1
    elif check_ssd(a, b, n_grid=n_grid):
        return 2
    else:
        return 0