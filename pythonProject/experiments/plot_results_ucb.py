import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import re

# ------------------------------
# CONFIG
# ------------------------------
saasbo=True
if saasbo:
    DATA_FOLDER = "dataframes/fb_prior/"
else:
    DATA_FOLDER = "dataframes"
FIG_FOLDER = "../figs"
SAVEFIG = True
PLOT = True

# ------------------------------
# LOAD ALL CSV FILES
# ------------------------------
all_files = glob.glob(os.path.join(DATA_FOLDER, "bo_*.csv"))

if not all_files:
    raise FileNotFoundError(f"No CSV files found in {DATA_FOLDER}")

df_list = []
for f in all_files:
    # Match filenames like bo_<function>_<method>_<seed>.csv
    # or bo_<function>_<method>_<seed>_alpha_cut.csv
    m = re.match(r"bo_(.*)_(.*)_(\d+)(_alpha_cut)?\.csv", os.path.basename(f))
    #if m:
    obj_name, method, seed, alpha_cut = m.groups()
    if alpha_cut:
        method += "_alphacut"  # append for consistency
    df = pd.read_csv(f)
    df["function"] = obj_name
    df["method"] = method
    df["seed"] = int(seed)
    df_list.append(df)
    #else:
    #    print(f"Skipping unrecognized file: {f}")


results_df = pd.concat(df_list, ignore_index=True)

print(f"Loaded {len(df_list)} CSV files, total rows: {len(results_df)}")
print("Functions:", results_df["function"].unique())
print("Methods:", results_df["method"].unique())

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


# Define colors, line styles, and markers for each method
plot_styles = {

    "Expected Improvement": {"color": "blue", "linestyle": "-", "marker": None},
    "Standard UCB": {"color": '#2D2D2D', "linestyle": "--", "marker": None},
    "Standard UCB (beta=2)": {"color": '#2D2D2D', "linestyle": ":", "marker": None},
    "Fully bayesian UCB": {"color": "red", "linestyle": "-", "marker": None},
    "Aggregation UCB": {"color": '#0000FF', "linestyle": "-", "marker": "v"},
    "Aggregation min": {"color": '#6666FF', "linestyle": "-", "marker": "s"},
    "Aggregation max": {"color": '#6666FF', "linestyle": "-", "marker": "D"},
    "Aggregation mean": {"color": "purple", "linestyle": "-", "marker": "^"},
    "Aggregation median": {"color": "purple", "linestyle": "-", "marker": "v"},
    "SD lowest variance": {"color": '#FFA500', "linestyle": "-", "marker": "D"},
    "SD Median": {"color": '#FFA500', "linestyle": "-", "marker": "s"},
    "SD best mean": {"color": '#CC8400', "linestyle": "-", "marker": "o"},
    "FB sanity": {"color": '#2D2D2D', "linestyle": "-", "marker": None},

}
# Define colors, line styles, and markers for each method
name_to_plot = {
    "Fully bayesian UCB": "FB-UCB",
    "Aggregation UCB": r'UCB $\mathcal{Q}$',
    "Aggregation min": r'min $\mathcal{Q}$',
    "Aggregation max": "Agg-max",
    "Aggregation mean": "Agg-mean",
    "Aggregation median": "Agg-median",
    "SD Median": "SD-median",
    "SD best mean": r'mu $\mathcal{X}$',
    "SD lowest variance": r'Var $\mathcal{X}$',
    "Expected Improvement": "EI",
    "Standard UCB": "UCB (β=1)",
    "Standard UCB (beta=2)": "UCB (β=2)",
    "FB sanity": "SAASBO",
}



ours = [

    "Aggregation UCB",

    "Aggregation min",

    #"SD Median",

    "SD best mean",
    "SD lowest variance",

]
baselines = [
    "FB sanity","Standard UCB",
    "Standard UCB (beta=2)",
    #"Fully bayesian UCB",
    #"Aggregation UCB",
    #"Aggregation min",
    #"Aggregation max",
    #"Aggregation mean",
    #"Aggregation median",
    #"SD lowest variance",
    #"SD Median",

    #"SD best mean",
]
ours_soft_revision = [
    #"Fully bayesian UCB_alphacut",

    "Aggregation UCB_alphacut",


    "Aggregation min_alphacut",

    #"Aggregation max_alphacut",
    #"Aggregation mean_alphacut",
    #"Aggregation median_alphacut",

    "SD lowest variance_alphacut",

    #"SD Median_alphacut",

    "SD best mean_alphacut",

]

func_to_plot = {
    "branin": "Branin 2D",
    "rosenbrock2d": "Rosenbrock 2D",
    "hartmann3": "Hartmann 3D",
    "griewank3": "Griewank 3D",
    "rosenbrock4d": "Rosenbrock 4D",
    "hartmann4": "Hartmann 4D",
    "rastrigin4": "Rastrigin 4D",
    "hartmann6": "Hartmann 6D",
    "sine1d": "Sine 1D",
    "ackley6": "Ackley 6D",
    "rastrigin8": "Rastrigin 8D",
    "griewank10": "Griewank 10D",
}
# PLOTTING: 2x2 grid with shared legend
# ------------------------------

soft_revision = True

methods_to_show = ours + baselines if not soft_revision else ours_soft_revision + baselines
if PLOT:
    func_names = [f for f in func_to_plot.keys() if f in results_df["function"].unique()]
    print("FUnc ",func_names)
    n_funcs = len(func_names)
    n_cols = 4 #n_funcs #-2
    n_rows = 1  # max 4 functions
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 2)) #, sharex=True, sharey=True)
    axes = axes.flatten()

    handles_labels = []
    for i in range(4): #range(len(func_names)): # - 2):
        func_name = func_names[i]
        func_df = results_df[results_df["function"] == func_name]
        print(i, func_df["method"].unique())


        ax = axes[i]
        ct = 0

        for method, g in func_df.groupby("method"):
            if method in methods_to_show:
                grouped = g.groupby("iteration")["regret"]
                mean_reg = grouped.mean()
                sem_reg = grouped.std() / np.sqrt(grouped.count())

                base_method = method.replace("_alphacut", "")
                style = plot_styles.get(base_method, {"color": "black", "linestyle": "-", "marker": None})
                linestyle = "--" if "_alphacut" in method else style["linestyle"]
                label = None if "_alphacut" in method else name_to_plot.get(base_method, base_method)

                num_markers = 5
                step = max(1, len(mean_reg) // num_markers)
                start = ct

                (line,) = ax.plot(
                    mean_reg.index,
                    mean_reg.values,
                    label=label,
                    color=style["color"],
                    marker=style["marker"],
                    markevery=(start, step),
                    linestyle=linestyle,
                )
                # Set all axis labels and ticks smaller
                ax.set_xlabel("Iteration", fontsize=8)
                #ax.set_ylabel("Log Regret", fontsize=6)  # if your y-axis is log-regret
                ax.tick_params(axis='both', which='major', labelsize=6)
                ax.tick_params(axis='both', which='minor', labelsize=6)
                ax.fill_between(
                    mean_reg.index,
                    mean_reg - sem_reg,
                    mean_reg + sem_reg,
                    color=style["color"],
                    alpha=0.2,
                )

                if label is not None:
                    handles_labels.append((line, label))
                ct += 5

        ax.set_title(func_to_plot.get(func_name, func_name), fontsize=8)
        ax.set_yscale("log")
        ax.grid(True)

    # Remove unused subplots if fewer than 4 functions
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Shared labels
    fig.supylabel("Log Regret", fontsize=8)
    # Reduce spacing between subplots
    fig.subplots_adjust(
        left=0.08,  # space on left
        right=0.95,  # space on right
        top=0.92,  # space on top
        bottom=0.08,  # space at bottom
        wspace=0.2,  # width space between subplots
        hspace=0.25  # height space between subplots
    )
    all_handles, all_labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)

    # Remove duplicates while preserving order
    unique_labels, unique_handles = [], []
    for lbl, hdl in zip(all_labels, all_handles):
        if lbl not in unique_labels:
            unique_labels.append(lbl)
            unique_handles.append(hdl)
    methods_to_plot = ours_soft_revision + baselines if soft_revision else ours + baselines
    # Build consistent sort order dictionary
    # Build sort order from methods_to_plot
    # Build sort order from internal method names
    sort_order = {m: i for i, m in enumerate(methods_to_plot)}

    # Build reverse mapping from plotted labels to method names
    label_to_method = {v: k for k, v in name_to_plot.items()}

    # Now sort
    sorted_pairs = sorted(
        zip(unique_labels, unique_handles),
        key=lambda x: sort_order.get(label_to_method.get(x[0], x[0]), len(sort_order))
    )

    sorted_labels, sorted_handles = zip(*sorted_pairs) if sorted_pairs else ([], [])

    print(sorted_pairs)
    # 3️⃣ Unpack sorted pairs
    sorted_labels = list(sorted_labels)
    sorted_handles = list(sorted_handles)

    fig.tight_layout()

    # Add legend above all subplots
    if not soft_revision:
        fig.legend(
            sorted_handles,
            sorted_labels,
            bbox_to_anchor=(0.97, 1.12),  # adjust for spacing as needed
            ncol=7,
            frameon=True,
            fontsize=9
        )


    # Save combined figure
    if SAVEFIG:
        if saasbo:
            func_folder = os.path.join(FIG_FOLDER, "bbob_fbprior")
        else:
            func_folder = os.path.join(FIG_FOLDER, "bbob")
        os.makedirs(func_folder, exist_ok=True)

        if soft_revision:
            fig_file = os.path.join(func_folder, "bbob_overview_2x2_soft_revision.pdf")
        else:
            fig_file = os.path.join(func_folder, "bbob_overview_2x2.pdf")
        fig.savefig(fig_file, facecolor="white", transparent=True, bbox_inches="tight")
        print(f"Saved combined figure to {fig_file}")

    plt.show()


