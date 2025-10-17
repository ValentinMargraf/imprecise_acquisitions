import torch
from botorch.models.fully_bayesian import FullyBayesianSingleTaskGP
from torch import Tensor
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP
from botorch.acquisition import (
    AcquisitionFunction,
    UpperConfidenceBound,
    ExpectedImprovement,
)
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.fit import fit_gpytorch_mll, fit_fully_bayesian_model_nuts
from gpytorch.mlls import ExactMarginalLogLikelihood
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from bbob_functions import rosenbrock4, hartmann3, hartmann6, branin2, rescale, ackley6, rastrigin8, griewank10, \
    griewank3, rastrigin4
from soft_revision import filter_models_by_alpha, compute_model_log_posteriors
from gaussian_process_models import pyro_gp_model, run_mcmc, build_gp_from_posterior, mixture_predict, get_mcmc
from acquisition_functions import MixtureUCB, sd_acqf_optimize, RiskAwareUCB, sd_acqf_optimize_median, RiskAwareEI, \
    MixtureEI
#from own_functions import OneDimSine
from gaussian_process_models import (
    pyro_gp_model,
    run_mcmc,
    build_gp_from_posterior,
    mixture_predict,
)
from acquisition_functions import (
    MixtureUCB,
    sd_acqf_optimize,
    RiskAwareUCB,
    sd_acqf_optimize_median,
)
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior


device = torch.device("cpu")
dtype = torch.float64


# -------------------------
# Objectives (all in [0,1]^d)
# -------------------------
noise_levels = {
    "branin2": 1e-3,
    "rosenbrock4": 1e-3,
    "hartmann3": 1e-3,
    "hartmann6": 1e-3,
    "ackley6": 1e-3,
    "rastrigin8": 1e-3,
    "griewank10": 1e-3,
    "rastrigin4": 1e-3,
    "griewank3": 1e-3,
}


objectives_done = {
    "hartmann3": {
        "func": lambda X: hartmann3(X, noise_std=noise_levels["hartmann3"]),
        "dim": 3,
        "fmin": -3.86278,
        "bounds": torch.stack([torch.zeros(3), torch.ones(3)])
    },
    "hartmann6": {
        "func": lambda X: hartmann6(X, noise_std=noise_levels["hartmann6"]),
        "dim": 6,
        "fmin": -3.32237,
        "bounds": torch.stack([torch.zeros(6), torch.ones(6)])
    },
    "rosenbrock4": {
        "func": lambda X: rosenbrock4(rescale(X, torch.tensor([[-5., -5., -5., -5.],
                                                                [ 5.,  5.,  5.,  5.]])),
                                       noise_std=noise_levels["rosenbrock4"]),
        "dim": 4,
        "fmin": 0.0,
        "bounds": torch.stack([torch.zeros(4), torch.ones(4)])
    },
    "branin2": {
        "func": lambda X: branin2(rescale(X, torch.tensor([[-5.,  0.],
                                                          [10., 15.]])),
                                 noise_std=noise_levels["branin2"]),
        "dim": 2,
        "fmin": 0.397887,
        "bounds": torch.stack([torch.zeros(2), torch.ones(2)])
    },
    "ackley6": {
        "func": lambda X: ackley6(rescale(X, torch.tensor([[-5., -5., -5., -5., -5., -5.],
                                                           [ 5.,  5.,  5.,  5.,  5.,  5.]])),
                                  noise_std=noise_levels["ackley6"]),
        "dim": 6,
        "fmin": 0.0,
        "bounds": torch.stack([torch.zeros(6), torch.ones(6)])
    },

    "rastrigin4": {
        "func": lambda X: rastrigin4(rescale(X, torch.tensor([[-5.12]*4,
                                                              [ 5.12]*4])),
                                     noise_std=noise_levels["rastrigin4"]),
        "dim": 4,
        "fmin": 0.0,
        "bounds": torch.stack([torch.zeros(4), torch.ones(4)])
    },
    "griewank3": {
        "func": lambda X: griewank3(rescale(X, torch.tensor([[-5.]*3,
                                                             [ 5.]*3])),
                                    noise_std=noise_levels["griewank3"]),
        "dim": 3,
        "fmin": 0.0,
        "bounds": torch.stack([torch.zeros(3), torch.ones(3)])
    },
"rastrigin8": {
    "func": lambda X: rastrigin8(rescale(X, torch.tensor([[-5.12]*8,
                                                          [ 5.12]*8])),
                                 noise_std=noise_levels["rastrigin8"]),
    "dim": 8,
    "fmin": 0.0,
    "bounds": torch.stack([torch.zeros(8), torch.ones(8)])
},

"griewank10": {
    "func": lambda X: griewank10(rescale(X, torch.tensor([[-600.]*10,
                                                          [ 600.]*10])),
                                 noise_std=noise_levels["griewank10"]),
    "dim": 10,
    "fmin": 0.0,
    "bounds": torch.stack([torch.zeros(10), torch.ones(10)])
},
}



# -------------------------
# BO loop (now generic)
# -------------------------
def run_bo(
    obj_func, X_init, Y_init, bounds, n_iter=5, num_gp_samples=16, method="Standard UCB", acqfunc="UCB", alpha_cut=False,
saasbo_priors=True
):
    X_train_scaled, Y_train = X_init.clone(), Y_init.clone()
    best_vals = [Y_train.min().item()]
    dim = X_train_scaled.shape[1]

    Y_mean = Y_train.mean()
    Y_std = Y_train.std(unbiased=True)
    Y_train_scaled = (Y_train - Y_mean) / Y_std


    bounds = bounds



    if saasbo_priors:
        print("saasbo")
        covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=dim,
            ))
    else:
        covar_module = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=X_train_scaled.shape[-1],
            use_rbf_kernel=True)

    for it in range(n_iter):
        #print("iter", it)
        if method == "Standard UCB":
            #gp = SingleTaskGP(X_train, Y_train)
            gp = SingleTaskGP(
                train_X=X_train_scaled,
                train_Y=Y_train_scaled,
                covar_module=covar_module,
                input_transform = None,  # already normalized
                outcome_transform = None,
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch)
            acq = UpperConfidenceBound(gp, beta=1.0, maximize=False)
            X_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=20, raw_samples=512)

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

            #print(f"Lengthscale: {lengthscale}")
            #print(f"Outputscale: {outputscale}")
            #print(f"Noise: {noise}")
            #print(f"Mean: {mean}")


        elif method == "Standard UCB (beta=2)":
            gp = SingleTaskGP(
                train_X=X_train_scaled,
                train_Y=Y_train_scaled,
                covar_module=covar_module,
                input_transform = None,  # already normalized
                outcome_transform = None,
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch)
            acq = UpperConfidenceBound(gp, beta=2.0, maximize=False)
            X_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=20, raw_samples=512)

        elif method == "Expected Improvement":
            gp = SingleTaskGP(
                train_X=X_train_scaled,
                train_Y=Y_train_scaled,
                covar_module=covar_module,
                input_transform=None,  # already normalized
                outcome_transform=None,
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                fit_gpytorch_mll(mll, optimizer=fit_gpytorch_mll_torch)

            acq = ExpectedImprovement(gp, best_f=Y_train_scaled.min().item(), maximize=False)
            X_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=20, raw_samples=512)

        else:



            if saasbo_priors:
                # Step 1: Fit the SaasFullyBayesianSingleTaskGP
                saas_gp = SaasFullyBayesianSingleTaskGP(
                    train_X=X_train_scaled,
                    train_Y=Y_train_scaled,
                    input_transform=None,  # already normalized
                    outcome_transform=None,
                )

                fit_fully_bayesian_model_nuts(
                    saas_gp,
                    warmup_steps=512,
                    num_samples=256,
                    thinning=16,
                    disable_progbar=True,
                )

                mcmc_samples = get_mcmc(saas_gp)
                num_gp_samples = mcmc_samples["mean0"].shape[0]  # number of posterior samples

                # Step 3: Build GP models manually from posterior samples
                gp_models = [
                    build_gp_from_posterior(
                        X_train_scaled,
                        Y_train_scaled,
                        {k: v[i] for k, v in mcmc_samples.items()},  # select the i-th sample
                        kernel = "matern52"
                    )
                    for i in range(num_gp_samples)
                ]
                #for each gp model compute marginal likelihood --> filter out
            else: # singletaskgp_priors:
                # 1. Run MCMC on scaled data
                posterior_samples = run_mcmc(X_train_scaled, Y_train_scaled, num_gp_samples=num_gp_samples, thinning=16,
                                             warmup=256)
                gp_models = [build_gp_from_posterior(X_train_scaled, Y_train_scaled, {k: v[i] for k, v in posterior_samples.items()})
                             for i in range(num_gp_samples)]

            if alpha_cut:
                log_post, details = compute_model_log_posteriors(
                    gp_models,
                    X_train_scaled,
                    Y_train_scaled,
                    extract_theta_fn=None,  # set if you want to include prior
                    log_prior_fn=None,  # set if available
                    use_prior=False  # set True if you provide log_prior_fn + extract_theta_fn
                )
                print("Log post", log_post)
                alpha = 0.05
                filtered_models, rels, keep_idx = filter_models_by_alpha(gp_models, log_post, alpha=alpha)

                print(f"Kept {len(filtered_models)} / {len(gp_models)} models (alpha={alpha})")
                gp_models = filtered_models

            if method == "FB sanity":
                # Sanity check: directly optimize UCB(λ=1) of the fitted SAASBO model
                acq = UpperConfidenceBound(saas_gp, beta=1.0, maximize=False) if acqfunc == "UCB" else ExpectedImprovement(
                    saas_gp, best_f=Y_train_scaled.min().item(), maximize=False)

                X_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=20, raw_samples=512)


            elif method == "Fully bayesian UCB":
                acq = MixtureUCB(gp_models, beta=1.0, maximize=False) if acqfunc == "UCB" else MixtureEI(gp_models, best_f=Y_train_scaled.min().item(), maximize=False)
                X_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=20, raw_samples=512)
            elif method == "Aggregation UCB":
                acq = RiskAwareUCB(gp_models, beta=1, lam=1.0, maximize=False, select_ucb=True) if acqfunc == "UCB" \
                    else RiskAwareEI(gp_models, best_f=Y_train_scaled.min().item(), lam=1.0, maximize=False, select_ucb=True)
                X_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=20, raw_samples=512)
            elif method == "Aggregation min":
                acq = RiskAwareUCB(gp_models, beta=1, lam=1.0, maximize=False, select_min=True) if acqfunc == "UCB" \
                    else RiskAwareEI(gp_models, best_f=Y_train_scaled.min().item(), lam=1.0, maximize=False, select_min=True)
                X_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=20, raw_samples=512)
            elif method == "Aggregation max":
                acq = RiskAwareUCB(gp_models, beta=1, lam=1.0, maximize=False, select_max=True) if acqfunc == "UCB" \
                    else RiskAwareEI(gp_models, best_f=Y_train_scaled.min().item(), lam=1.0, maximize=False, select_max=True)
                X_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=20, raw_samples=512)
            elif method == "Aggregation mean":
                acq = RiskAwareUCB(gp_models, beta=1, lam=0.0, maximize=False, select_ucb=True) if acqfunc == "UCB" \
                    else RiskAwareEI(gp_models, best_f=Y_train_scaled.min().item(), lam=0.0, maximize=False, select_ucb=True)
                X_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=20, raw_samples=512)
            elif method == "Aggregation median":
                acq = RiskAwareUCB(gp_models, beta=1, lam=1.0, maximize=False, select_median=True) if acqfunc == "UCB" \
                    else RiskAwareEI(gp_models, best_f=Y_train_scaled.min().item(), lam=1.0, maximize=False, select_median=True)
                X_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=20, raw_samples=512)

            elif method == "SD lowest variance":
                X_next = sd_acqf_optimize(gp_models, bounds, maximize=False, beta=1.0, num_restarts=20, raw_samples=512,select_lowest_variance=True)
            elif method == "SD best mean":
                X_next = sd_acqf_optimize(gp_models, bounds, maximize=False, beta=1.0, num_restarts=20, raw_samples=512,
                                          select_lowest_variance=False)
            elif method == "SD Median":
                X_next = sd_acqf_optimize_median(gp_models, bounds, maximize=False, beta=1.0, num_restarts=20, raw_samples=512,best_f=Y_train_scaled.min().item())

        # Evaluate next point in original units
        y_next = obj_func(X_next).view(-1, 1)

        # Append to original Y (for metrics)
        Y_train = torch.cat([Y_train, y_next], dim=0)

        # Standardize for GP fitting
        Y_train_scaled = (Y_train - Y_train.mean()) / Y_train.std(unbiased=True)


        # Append new X
        X_train_scaled = torch.cat([X_train_scaled, X_next], dim=0)

        # Track best observed value in original units
        best_vals.append(Y_train.min().item())
        print("Best val", Y_train.min().item())

    return best_vals


import pandas as pd
import os
import sys

# -------------------------
# Main experiment
# -------------------------
if __name__ == "__main__":
    n_iters = {1: 50, 2: 77, 3: 88, 4: 100, 6: 134, 8: 140, 10: 160}
    initial_samples = {dim: int(0.2 * n_iters[dim]) for dim in n_iters}

    num_objectives = len(objectives.items())
    print("num_objectives", num_objectives)
    singletaskgp_priors = False
    saasbo_priors = True

    num_gp_samples = 16
    n_configs = 13
    n_seeds = 20

    job_index = int(sys.argv[1])  # SLURM_ARRAY_TASK_ID
    acqfunc = "UCB"
    # acqfunc = "EI"


    alpha_cut = True


    # Compute seed, config, and objective based on job_index
    seed = job_index % n_seeds
    config = (job_index // n_seeds) % n_configs
    objective_idx = (job_index // (n_seeds * n_configs)) % num_objectives

    print("objective_idx", objective_idx)
    print("config, ", config)
    print("seed, ", seed)
    torch.manual_seed(seed)

    configs_dict = {
        0: "Expected Improvement",
        1: "Standard UCB",
        2: "Standard UCB (beta=2)",
        3: "Fully bayesian UCB",
        4: "Aggregation UCB",
        5: "Aggregation min",
        6: "Aggregation max",
        7: "Aggregation mean",
        8: "Aggregation median",
        9: "SD lowest variance",
        10: "SD Median",
        11: "SD best mean",
        12: "FB sanity",
    }

    # Get objective name and info
    obj_name = list(objectives.keys())[objective_idx]
    obj_info = objectives[obj_name]
    func, dim, fmin, bounds = obj_info["func"], obj_info["dim"], obj_info["fmin"], obj_info["bounds"]
    method = configs_dict[config]

    if saasbo_priors:
        results_file = f"dataframes/fb_prior/bo_{obj_name}_{method}_{seed}"
    else:
        results_file = f"dataframes/bo_{obj_name}_{method}_{seed}"
    if alpha_cut:
        results_file += "_alpha_cut"
    else:
        results_file += ".csv"

    # Set n_init and n_iter dynamically
    n_iter = n_iters[dim]
    n_init = initial_samples[dim]

    # Load existing results if available
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        done = set(zip(results_df["method"], results_df["seed"]))
    else:
        results_df, done = pd.DataFrame(), set()


    # Skip if this combination is already done
    if (method, seed) in done:
        print(f"Skipping {obj_name}, {method}, seed={seed}")
    else:
        print(f"Running {obj_name}, {method}, seed={seed}")
        # --- run BO here ---


    # initial design in [0,1]^d
    X_init = torch.rand(n_init, dim, dtype=dtype, device=device)
    Y_init = func(X_init).unsqueeze(-1)

    best_values = run_bo(func, X_init, Y_init, bounds, n_iter, num_gp_samples, method=method, acqfunc=acqfunc)

    # record results
    df_records = []
    for t, val in enumerate(best_values):
        df_records.append({
            "function": obj_name,
            "method": method,
            "seed": seed,
            "iteration": t,
            "best_value": val,
            "regret": val - fmin,
        })

    new_df = pd.DataFrame(df_records)
    results_df = pd.concat([results_df, new_df], ignore_index=True)
    results_df.to_csv(results_file, index=False)