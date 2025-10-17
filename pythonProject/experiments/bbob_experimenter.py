from __future__ import annotations
import os
import sys
import torch
import pandas as pd

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor

from run_bbob import run_bo
from bbob_functions import rosenbrock4, hartmann3, hartmann6, branin2, rescale, ackley6, rastrigin8, griewank10, \
    griewank3, rastrigin4
from botorch.utils.sampling import draw_sobol_samples

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


objectives = {
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


device = torch.device("cpu")
dtype = torch.float64


def run_config(config: dict, result_processor: ResultProcessor, custom_config: dict):
    """
    Runs a single Bayesian Optimization experiment given a configuration.
    """
    seed = int(config["seed"])
    objective = config["objective"]
    acq_func = config["acq_func"]
    method = config["method"]
    alpha_cut = bool(int(config["alpha_cut"]))
    saasbo_priors = bool(int(config["saasbo_priors"]))
    # Set random seed
    torch.manual_seed(seed)

    # Informative log
    print(f"[INFO] Running method '{method}' on objective '{objective}' with seed {seed} "
          f"| Acquisition: {acq_func} | Alpha-cut: {alpha_cut} | SaasBO priors: {saasbo_priors}")
    # Map config index to method name
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

    # Select objective
    obj_info = objectives[objective]
    func, dim, fmin, bounds = obj_info["func"], obj_info["dim"], obj_info["fmin"], obj_info["bounds"]

    # BO loop settings
    n_iters = {1: 50, 2: 77, 3: 88, 4: 100, 6: 134, 8: 140, 10: 160}
    initial_samples = {dim: int(0.2 * n_iters[dim]) for dim in n_iters}
    n_iter = n_iters[dim]
    n_init = initial_samples[dim]
    num_gp_samples = 16


    # Draw Sobol samples
    X_init = draw_sobol_samples(
        bounds=bounds.to(dtype=dtype, device=device),
        n=n_init,  # number of initial points
        q=1,  # one point at a time
        seed=seed
    ).squeeze(1)  # shape [n_init, dim]

    # Evaluate the objective at Sobol points
    Y_init = func(X_init).unsqueeze(-1)

    # Run BO
    best_values = run_bo(func, X_init, Y_init, bounds, n_iter=n_iter, num_gp_samples=num_gp_samples, method=method, acqfunc=acq_func, alpha_cut=alpha_cut, saasbo_priors=saasbo_priors)

    best_values_list = [float(val) for val in best_values]
    regrets = [float(val - fmin) for val in best_values]

    #records = [{"best_value": val, "regret": r} for val, r in zip(best_values_list, regrets)]
    #result_processor.process_logs({"results": records})
    result_processor.process_logs({
        'results': {
            'best_value': str(best_values_list),
            'regret': str(regrets),
            'fmin': float(fmin),
        }
    })



def run_parallel(variable: str, run_setup=False, reset_experiments=False):
    experimenter = PyExperimenter(
        experiment_configuration_file_path="configs/exp_bbob_fb_prior.yml",  # update path if needed
        database_credential_file_path="configs/db_conf.yml",
        name=variable,
        use_codecarbon=False
    )

    if run_setup:
        if reset_experiments:
            experimenter.reset_experiments( 'error', 'running')
        else:
            experimenter.fill_table_from_combination(
                parameters={
                    "seed": list(range(20)),
                    "objective": [
                            "branin2",
                            "rosenbrock4",
                            "hartmann3",
                            "hartmann6",
                            "ackley6",
                            "rastrigin8",
                            "griewank10",
                            "rastrigin4",
                            "griewank3"
                        ],
                    "saasbo_priors": 1,
                    "acq_func": "EI",
                    "method": [
                        "Expected Improvement",
                        #"Standard UCB",
                        #"Standard UCB (beta=2)",
                        "Fully bayesian UCB",
                        "Aggregation UCB",
                        "Aggregation min",
                        "Aggregation max",
                        "Aggregation mean",
                        "Aggregation median",
                        "SD lowest variance",
                        "SD Median",
                        "SD best mean",
                        "FB sanity"],
                    "alpha_cut": [0],#1]

                }
            )
    else:
        experimenter.execute(run_config, max_experiments=-1)



if __name__ == "__main__":
    variable_job_id = str(sys.argv[1])
    variable_task_id = str(sys.argv[2])
    variable = f"{variable_job_id}_{variable_task_id}"
    run_parallel(variable, run_setup=True, reset_experiments=True)
