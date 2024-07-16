import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# plt.rc('pgf', texsystem='pdflatex')
plt.style.use(['classic'])
plt.rcParams["figure.figsize"] = (17,17)
import os
import pandas as pd
from collections import defaultdict
import numpy as np

mode="classification"
datasets={"classification": ["gaussianPiecewiseConstant", "gaussianLinear","gaussianNonLinearAdditive"]}
metrics = ["infidelity"]
results_dir="results"
output_dir=f"plots/{mode}/error_bands/"
csv = "csv"
exp_names=["multi-experiment"]

# Important initializations
rhos = [0.0, 0.25, 0.5, 0.75, 0.99]
explainers = ["kernelshap", "lime", "maple", "l2x", "breakdown"]
explainer_mappings = {
    "kernelshap": "KernelSHAP",
    "lime": "LIME",
    "maple":"Maple",
    "l2x": "L2X",
    "breakdown": "Breakdown"
}
# models
models = ["XGB", "SVM"]

def collect_data(file_path, rho):
    lines = open(file_path, 'r').readlines()
    for idx, line in enumerate(lines):
        if line.strip().split(",")[0] in models:
            model = line.strip().split(",")[0]
            model_perfs[model][rho] = line.strip().split(",")[1:]
            df = pd.read_csv(file_path, skiprows=idx+1, nrows=len(explainers))
            df.columns.values[0] = "explainer"
            scores[model][rho].append(df)

for metric in metrics:
    num_models, num_datasets = len(models), len(datasets[mode])
    fig, axs = plt.subplots(figsize=(5.0*num_models, 4.0*num_datasets), nrows=num_datasets, ncols=num_models)

    for data_idx, dataset in enumerate(datasets[mode]):
        scores = {model: defaultdict(list) for model in models}
        model_perfs = {model: defaultdict(list) for model in models}

        for rho in rhos:
            for exp_name in exp_names:
                #file_path = os.path.join(results_dir, mode, exp_name, csv, f"{dataset}_{rho}.csv")
                file_path = f"{dataset}_{rho}.csv"
                collect_data(file_path, rho)

        for idx, model in enumerate(models):
            for explainer in explainers:
                values = [[] for _ in range(len(exp_names))]
                for exp in range(len(exp_names)):
                    values[exp] = [scores[model][r][exp].query(f'explainer=="{explainer}"')[metric] for r in rhos]
                values = np.squeeze(np.array(values))
                means = np.mean(values, axis=0)
                mins = np.min(values, axis=0)
                maxs = np.max(values, axis=0)
                if (num_models==1):
                    if (num_datasets == 1):
                        axs.plot(rhos, values, linewidth=2,  label = explainer)
                        #axs.fill_between(rhos, mins, maxs, alpha=0.2)
                    else:
                        axs[data_idx].plot(rhos, values, linewidth=2,  label = explainer)
                        axs[data_idx].fill_between(rhos, mins, maxs, alpha=0.2)
                else:
                    if (num_datasets == 1):
                        axs[idx].plot(rhos, values, linewidth=2,  label = explainer)
                        #axs[idx].fill_between(rhos, mins, maxs, alpha=0.2)
                    else:
                        axs[data_idx][idx].plot(rhos, values, linewidth=2,  label = explainer)
                        #axs[data_idx][idx].fill_between(rhos, mins, maxs, alpha=0.2)
            if (num_models==1):
                if (num_datasets == 1):
                    axs.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                else:
                    axs[data_idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            else:
                if (num_datasets == 1):
                    axs[idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                else:
                    axs[data_idx][idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            perfs = [float(model_perfs[model][r][1]) for r in rhos]
            if (num_models==1):
                if (num_datasets == 1):
                    axs.set_xlabel("Rho")
                else:
                    axs[data_idx].set_xlabel("Rho")
            else:
                if (num_datasets == 1):
                    axs[idx].set_xlabel("Rho")
                else:
                    axs[data_idx][idx].set_xlabel("Rho")
            if idx == 0:
                if (num_models==1):
                    if (num_datasets == 1):
                        axs.set_ylabel(metric, fontsize=14)
                    else:
                        axs[data_idx].set_ylabel(metric, fontsize=14)
                else:
                    if (num_datasets == 1):
                        axs[idx].set_ylabel(metric, fontsize=14)
                    else:
                        axs[data_idx][idx].set_ylabel(metric, fontsize=14)
            if data_idx == 0:
                if (num_models==1):
                    if (num_datasets == 1):
                        axs.set_title(model)
                    else:
                        axs[data_idx].set_title(model)
                else:
                    if (num_datasets == 1):
                        axs[idx].set_title(model)
                    else:
                        axs[data_idx][idx].set_title(model)
            if model == "XGB" or model == "SVM" or model == "RFC":
                if (num_models==1):
                    if (num_datasets == 1):
                        lines, labels = axs.get_legend_handles_labels()
                    else:
                        lines, labels = axs[data_idx].get_legend_handles_labels()
                else:
                    if (num_datasets == 1):
                        lines, labels = axs[idx].get_legend_handles_labels()
                    else:
                        lines, labels = axs[data_idx][idx].get_legend_handles_labels()
                lines2, labels2 = [] ,[]
    fig.subplots_adjust(bottom=0.12)
    fig.legend(lines + lines2, [explainer_mappings[a] for a in labels + labels2], frameon=True, shadow=True, loc="lower center", bbox_to_anchor=(0.43, 0), ncol=len(explainers), fontsize=14)
    fig.set_size_inches(20, 12)
    fig.set_facecolor('white')
    plt.show()
    plt_save_path = os.path.join(output_dir, f"{metric}_new_all.pdf")
    if not os.path.exists(os.path.dirname(plt_save_path)):
        os.makedirs(os.path.dirname(plt_save_path))
    #fig.tight_layout()
    plt.savefig(plt_save_path, bbox_inches='tight')
    #plt.clf()
    #plt.cla()