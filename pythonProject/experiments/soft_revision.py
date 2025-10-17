import torch
import math
from typing import Callable, Iterable, List, Dict, Any

EPS = 1e-6

def compute_marginal_log_likelihood_from_gp(gp, X_train, Y_train, jitter=1e-6):
    """
    Compute log p(y | theta) for a SingleTaskGP `gp` built with hyperparameters theta.
    Uses the closed-form Gaussian marginal likelihood:
      -0.5 * y^T K^{-1} y - 0.5 * log|K| - (n/2) * log(2*pi)
    where K = K_theta(X,X) + sigma^2 I.
    Args:
        gp: a SingleTaskGP instance whose covar_module and likelihood are already set.
        X_train: (n x d) tensor (same used to build gp)
        Y_train: (n x 1) tensor
        jitter: numerical jitter added to diagonal for stability
    Returns:
        scalar tensor: marginal log likelihood
    """
    device = X_train.device
    dtype = X_train.dtype
    n = X_train.shape[0]

    gp.eval()
    gp.likelihood.eval()
    with torch.no_grad():
        # Evaluate kernel -> returns a LazyTensor for many kernels; call .evaluate() for dense matrix
        K = gp.covar_module(X_train).evaluate()  # shape: n x n
        # Get noise (likelihood.noise is either scalar or tensor)
        noise = gp.likelihood.noise
        # Ensure noise is broadcastable to (n,n)
        if noise.numel() == 1:
            noise_val = noise.reshape(1).to(dtype=dtype, device=device).item()
            K = K + (noise_val + jitter) * torch.eye(n, device=device, dtype=dtype)
        else:
            # per-observation noise
            noise_vec = noise.reshape(-1).to(dtype=dtype, device=device)
            K = K + torch.diag(noise_vec + jitter)

        # Convert to dense if still lazy (evaluate did this) and ensure symmetric
        # Cholesky (stable) decomposition
        try:
            L = torch.linalg.cholesky(K)
        except RuntimeError:
            # try adding more jitter if decomposition fails
            jitter2 = 1e-4
            K = K + jitter2 * torch.eye(n, device=device, dtype=dtype)
            L = torch.linalg.cholesky(K)

        y = Y_train.reshape(-1).to(dtype=dtype, device=device)  # shape n
        # solve L v = y  -> v = L^{-1} y
        v = torch.cholesky_solve(y.unsqueeze(-1), L) if hasattr(torch, "cholesky_solve") else torch.linalg.solve(L, y.unsqueeze(-1))
        # cholesky_solve returns (n,1)
        quad = 0.5 * (y.unsqueeze(0) @ v).squeeze()   # scalar
        logdet = torch.sum(torch.log(torch.diag(L)))
        const = 0.5 * n * math.log(2 * math.pi)

        log_marginal = -(quad + logdet + const)
        return log_marginal.squeeze()

def compute_model_log_posteriors(
    gp_models: List[Any],
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    extract_theta_fn: Callable[[Any], Dict[str, torch.Tensor]] | None = None,
    log_prior_fn: Callable[[Dict[str, torch.Tensor]], float] | None = None,
    use_prior: bool = True,
):
    """
    For each gp in gp_models compute ell_j = log p(y | theta_j) + log p(theta_j) (if available).
    - extract_theta_fn(gp) -> dict of raw theta values (optional)
    - log_prior_fn(theta_dict) -> scalar log p(theta) (optional)
    Returns:
      log_posteriors: Tensor shape (m,)
      details: list of dicts with entries {'log_lik':..., 'log_prior':..., 'ell':...}
    """
    device = X_train.device
    log_liks = []
    details = []
    for gp in gp_models:
        log_lik = compute_marginal_log_likelihood_from_gp(gp, X_train, Y_train)
        log_prior = 0.0
        if use_prior and log_prior_fn is not None and extract_theta_fn is not None:
            theta_dict = extract_theta_fn(gp)  # you must implement this for your hyperprior
            log_prior = float(log_prior_fn(theta_dict))
        ell = float(log_lik) + float(log_prior)
        log_liks.append(float(log_lik))
        details.append({"log_lik": float(log_lik), "log_prior": float(log_prior), "ell": ell})
    log_post = torch.tensor([d["ell"] for d in details], device=device, dtype=X_train.dtype)
    return log_post, details


def filter_models_by_alpha(
        gp_models: List[Any],
        log_posteriors: torch.Tensor,
        alpha: float = 0.1,
        min_models: int = 8,
        fallback_alpha: float = 0.05,
):
    """
    Filter gp_models by relative posterior (unnormalized) w.r.t. max.
      rel_j = exp(ell_j - max_ell)
    Keep models with rel_j >= alpha.
    If too few models survive, keep at least `min_models` best models.
    Returns filtered_models, rels (numpy array), keep_idx (list)
    """
    device = log_posteriors.device
    max_ell = torch.max(log_posteriors)
    rel = torch.exp(log_posteriors - max_ell)  # in (0,1]
    print("rel", rel)

    keep_mask = rel >= alpha
    keep_idx = torch.nonzero(keep_mask).squeeze(-1).tolist()

    # fallback if too few models
    if len(keep_idx) < min_models:
        # sort indices by descending rel
        sorted_idx = torch.argsort(rel, descending=True)
        keep_idx = sorted_idx[:min_models].tolist()
        print(f"Too few models >= alpha; keeping top {min_models} models")

    filtered_models = [gp_models[i] for i in keep_idx]
    rels = rel.cpu().numpy()

    return filtered_models, rels, keep_idx
