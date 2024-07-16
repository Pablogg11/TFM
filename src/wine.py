import pandas as pd
import numpy as np
import logging
import json
import os

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

from synthetic_datasets import GaussianLinearRegression, GaussianLinearBinary
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from src import datasets, explainer, experiments, metric, model, parse_utils
from sklearn.metrics import mean_squared_error

mode = "classification"
results_dir = f"results/{mode}/ibk/"

models = ["mlp", "dtree", "rfc", "xgb", "svm"]

# explainers = ["shap", "brutekernelshap", "kernelshap", "breakdown", "ibreakdown", "lime", "maple", "l2x", "axom"]
explainers = ["ibreakdown"]

metrics = ["roar", "faithfulness", "monotonicity", "shapley", "infidelity"]

df = pd.read_csv('data/WineQT.csv', sep=',')

X = df.drop('quality', axis=1)
X = preprocessing.StandardScaler().fit(X).transform(X)
y = df['quality']

X, X_val, y, y_val = train_test_split(X, y, test_size=0.01, random_state=7)
knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X,y)
mean = np.mean(X, axis=0)
cov = np.cov(X, rowvar=False)

data_generator = datasets.Data("gaussianLinear", mode=mode, mu=repr(mean), dim=len(mean), noise=0.01, 
                                            sigma=repr(cov), weight=repr(np.ones(len(mean))))


def make_experiment_with_dataset(dataset, models, explainers, metrics):
    models = [
        model.Model(mod, mode)
        for mod in models
    ]
    explainers = [
        explainer.Explainer(expl) for expl in explainers
    ]
    metrics = [metric.Metric(metr) for metr in metrics]
    return experiments.Experiment(dataset, models, explainers, metrics)


experiment = make_experiment_with_dataset(data_generator, models, explainers, metrics)
results = experiment.get_results()
logging.info(f"\nExperiment results : {json.dumps(results, indent=4)}")

parse_utils.save_experiment(experiment, os.path.join(results_dir, "checkpoints"), "na")
parse_utils.save_results(results, results_dir)
parse_utils.save_results_csv(results, results_dir)