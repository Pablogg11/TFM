from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost

valid_models = {
    "regression": {
        "dataset": lambda : "dataset",
        "lr": LinearRegression,
        "mlp": MLPRegressor,
        "dtree": DecisionTreeRegressor, 
        "xgb": xgboost.XGBRegressor,
        "svm": SVR
    },
    "classification": {
        "dataset": lambda : "dataset",
        "lr": LinearRegression,
        "mlp": MLPClassifier,
        "dtree": DecisionTreeClassifier,
        "rfc": RandomForestClassifier,
        "xgb": xgboost.XGBClassifier,
        "svm": LinearSVC
    },
}


class Model:
    def __init__(self, name, mode, **kwargs):
        if name not in valid_models[mode].keys():
            raise NotImplementedError(
                f"This model is not supported at the moment. Models supported are: {list(valid_models[mode].keys())}"
            )
        self.name = name
        self.mode = mode
        self.model = valid_models[mode][name](**kwargs)
        if self.model == "svm":
            self.model.set_params(n_estimators=50)
        if self.model == "dataset":
            return
        self.predict = self.model.predict
        if self.model.fit:
            self.train = self.model.fit
