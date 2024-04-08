import random
import sys
import traceback

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import shap

class Axom:
    def __init__(self, model, X):
        self.model = model
        self.train_set = X
    
    def explain(self,  test_set):
        labels = self.model.predict(test_set.values).astype(int)
        shape = test_set.values.shape
        weak_explainers = []
        exps = [[] for _ in range(shape[0])]
        for k, weak in enumerate(self.model.estimators_):
            weak_explainer = shap.Explainer(weak)
            weak_explainers.append((weak, weak_explainer))
            weak_labels = weak.predict(test_set.values)
            for i in range(len(test_set)):
                if weak_labels[i] != labels[i]:
                    continue
                weak_exp = weak_explainer(test_set.loc[i]).values[:, labels[i]]
                exps[i].append(weak_exp)

        axom_shap = []
        for i, sample in enumerate(test_set.iloc):
            g_x_i = np.mean(exps[i], axis=0)
            axom_shap.append(g_x_i)
            
        return np.array(axom_shap)