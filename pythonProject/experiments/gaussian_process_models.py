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
DEBUG = False

# -------------------------------------------------
# 1. Helpers for normalization / standardization
# -------------------------------------------------
def normalize_inputs(X):
    X_min, X_max = X.min(dim=0).values, X.max(dim=0).values
    X_scaled = (X - X_min) / (X_max - X_min + 1e-12)
    #print("X_scaled", X_scaled[0])
    #print("X", X)
    return X_scaled, X_min, X_max

def standardize_outputs(Y):
    Y_mean, Y_std = Y.mean(), Y.std()
    Y_std = Y_std if Y_std > 0 else torch.tensor(1.0, device=Y.device, dtype=Y.dtype)
    Y_scaled = (Y - Y_mean) / Y_std
    return Y_scaled, Y_mean, Y_std

# %%
# -------------------------
# Mixture GP predictions
# -------------------------
def mixture_predict(gp_models, X: Tensor):
    mus, vars_ = [], []

    for idx,gp in enumerate(gp_models):
        post = gp.posterior(X)
        if isinstance(post, tuple):
            mean, var = post
        else:
            mean = post.mean.squeeze(-1)
            var = post.variance.squeeze(-1)
        #mean = post.mean.squeeze(-1)
        #var = post.variance.squeeze(-1)
        mus.append(mean)
        vars_.append(var)
    mus = torch.stack(mus)
    vars_ = torch.stack(vars_)
    sigmas = torch.sqrt(torch.clamp(vars_, min=1e-12))
    return mus, vars_, sigmas

# -------------------------------------------------
# 2. Pyro model for MCMC
# -------------------------------------------------
def pyro_gp_model(X, Y):
    n, d = X.shape
    dtype = X.dtype

    # --- Priors for hyperparameters ---
    mu = torch.sqrt(torch.tensor(2.0)) + torch.log(torch.sqrt(torch.tensor(float(d), dtype=dtype)))
    sigma = torch.sqrt(torch.tensor(3.0, dtype=dtype))
    lengthscale = pyro.sample("lengthscale", dist.LogNormal(mu * torch.ones(d, dtype=dtype), sigma * torch.ones(d, dtype=dtype)))

    outputscale = pyro.sample(
        "outputscale",
        dist.Normal(torch.tensor(1, dtype=dtype), torch.tensor(1e-3, dtype=dtype))
    )
    noise = pyro.sample("noise", dist.LogNormal(torch.tensor(-4.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)))

    # Since Y is standardized, prior mean ~ 0
    mean0 = pyro.sample("mean0", dist.Normal(torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)))

    # --- Covariance ---
    diff = X[:, None, :] - X[None, :, :]
    K = outputscale * torch.exp(-0.5 * ((diff / lengthscale) ** 2).sum(-1))
    cov = K + (noise ** 2 + 1e-8) * torch.eye(n, dtype=dtype)

    pyro.sample("Y", dist.MultivariateNormal(mean0 * torch.ones(n, dtype=dtype), cov), obs=Y.squeeze())



# -------------------------------------------------
# 3. Run MCMC on scaled data
# -------------------------------------------------
def run_mcmc(X_train, Y_train, num_gp_samples=16, thinning=16, warmup=512):
    total_samples = num_gp_samples * thinning
    # Normalize and standardize
    # --- Sanity check: are X and Y standardized? ---
    print("==== Data Standardization Check ====")
    print(f"X_train mean per dimension: {X_train.mean(0)}")
    print(f"X_train std per dimension:  {X_train.std(0, unbiased=False)}")
    print(f"Y_train mean: {Y_train.mean().item():.4f}")
    print(f"Y_train std:  {Y_train.std(unbiased=False).item():.4f}")
    print("====================================")

    nuts = NUTS(
        pyro_gp_model,
        jit_compile=False,
        full_mass=True,
        ignore_jit_warnings=True,
        max_tree_depth=6,
    )


    mcmc = MCMC(
        nuts,
        warmup_steps=warmup,
        num_samples=total_samples,
        disable_progbar=False,
    )
    mcmc.run(X_train, Y_train.squeeze())
    posterior_samples = mcmc.get_samples()

    thinned_samples = {k: v[::thinning] for k, v in posterior_samples.items()}

    return thinned_samples


def get_mcmc(gp):
    # Step 2: Extract MCMC samples from the fitted model
    # Note: SaasFullyBayesianSingleTaskGP has a `state_dict()` containing the samples
    state_dict = gp.state_dict()
    #print("state ", state_dict)
    def softplus_transform(x):
        return torch.nn.functional.softplus(x)

    # Extract MCMC samples (raw, unconstrained values)
    mcmc_samples_raw = {
        "mean0": state_dict["mean_module.raw_constant"],
        "lengthscale": state_dict["covar_module.base_kernel.raw_lengthscale"],
        "outputscale": state_dict["covar_module.raw_outputscale"],
    }

    # Include noise if available
    if "likelihood.noise_covar.raw_noise" in state_dict:
        mcmc_samples_raw["noise"] = state_dict["likelihood.noise_covar.raw_noise"]

    # Transform unconstrained samples into physical space
    mcmc_samples = {
        "mean0": mcmc_samples_raw["mean0"],  # mean can stay as-is (can be positive or negative)
        "lengthscale": softplus_transform(mcmc_samples_raw["lengthscale"]),
        "outputscale": softplus_transform(mcmc_samples_raw["outputscale"]),
        "noise": softplus_transform(mcmc_samples_raw["noise"]) if "noise" in mcmc_samples_raw else None,
    }

    return mcmc_samples

# -------------------------------------------------
# 4. Build GP from posterior sample (scaled domain)
# -------------------------------------------------
def build_gp_from_posterior(X_train, Y_train, posterior_sample, kernel="rbf"):
    #X_train_scaled, _, _ = normalize_inputs(X_train)
    #Y_train_scaled, _, _ = standardize_outputs(Y_train)

    device = X_train.device
    dtype = X_train.dtype
    d = X_train.shape[1]

    noise_value = posterior_sample["noise"].to(dtype=dtype, device=device)

    #print("Noise value:", noise_value)
    #print("mean ", posterior_sample["mean0"])
    #print("lengthscale ", posterior_sample["lengthscale"])
    #print("out ", posterior_sample["outputscale"].to(dtype=dtype, device=device))

    mean_module = ConstantMean()
    mean_module.initialize(constant=posterior_sample["mean0"].to(dtype=dtype, device=device))
    mean_module.constant.requires_grad_(False)

    if kernel == "rbf":
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=d))
    elif kernel == "matern52":
        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))
    lengthscale = posterior_sample["lengthscale"].to(dtype=dtype, device=device).flatten()
    covar_module.base_kernel.initialize(lengthscale=lengthscale)
    covar_module.initialize(outputscale=posterior_sample["outputscale"].to(dtype=dtype, device=device))


    noise_value = torch.clamp(noise_value, min=1e-4)
    likelihood = GaussianLikelihood()
    likelihood.initialize(noise=noise_value)

    gp = SingleTaskGP(
        X_train,
        Y_train,
        mean_module=mean_module,
        input_transform=None,       # already normalized
        outcome_transform=None,     # already standardized
    )
    gp.covar_module = covar_module
    gp.likelihood = likelihood
    gp.eval()
    gp.likelihood.eval()
    # Lengthscale
    if hasattr(gp.covar_module, "base_kernel"):  # e.g., ScaleKernel
        lengthscale = gp.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
        outputscale = gp.covar_module.outputscale.detach().cpu().numpy()
    else:  # e.g., just RBFKernel
        lengthscale = gp.covar_module.lengthscale.detach().cpu().numpy()
        outputscale = None

    # Noise
    noise = gp.likelihood.noise.detach().cpu().numpy()

    # Mean
    mean = gp.mean_module.constant.detach().cpu().numpy() if hasattr(gp.mean_module, "constant") else None
    if DEBUG:
        print(f"Lengthscale: {lengthscale}")
        print(f"Outputscale: {outputscale}")
        print(f"Noise: {noise}")
        print(f"Mean: {mean}")
        print("covar module", covar_module)
    return gp



# -------------------------------------------------
# 5. Untransform predictions if needed
# -------------------------------------------------
def unnormalize_inputs(X_scaled, X_min, X_max):
    return X_scaled * (X_max - X_min) + X_min

def unstandardize_outputs(Y_scaled, Y_mean, Y_std):
    return Y_scaled * Y_std + Y_mean

# -------------------------
# Pyro fully Bayesian GP model
# -------------------------
def score_bo(X, Y):
    n, d = X.shape

    # Priors
    lengthscale = pyro.sample(
        "lengthscale",
        dist.Gamma(torch.tensor(3.0, dtype=dtype), torch.tensor(6.0, dtype=dtype)).expand([d]).to_event(1)
    )
    outputscale = pyro.sample(
        "outputscale",
        dist.Gamma(torch.tensor(2.0, dtype=dtype), torch.tensor(0.15, dtype=dtype))
    )
    noise_var = pyro.sample(
        "noise",
        dist.Gamma(torch.tensor(1.1, dtype=dtype), torch.tensor(0.05, dtype=dtype))
    )
    mean_c = pyro.sample(
        "mean0",
        dist.Normal(torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype))
    )

    # Kernel matrix
    diff = X[:, None, :] - X[None, :, :]
    K = outputscale * torch.exp(-0.5 * ((diff / lengthscale) ** 2).sum(-1))

    # Add noise variance to diagonal
    cov = K + (noise_var + 1e-8) * torch.eye(n, dtype=dtype)

    # GP likelihood
    pyro.sample(
        "Y",
        dist.MultivariateNormal(mean_c * torch.ones(n, dtype=dtype), cov),
        obs=Y.squeeze()
    )
