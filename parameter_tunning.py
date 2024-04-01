import os
import optuna
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
import numpy as np
from pyarrow import csv
import json


#data
PROJECT_PATH = os.path.dirname(__file__)
DATA_PATH = PROJECT_PATH + "/data/to_train/"

X_train = csv.read_csv(DATA_PATH + "X_train.csv")
Y_train = np.genfromtxt(DATA_PATH + "Y_train.csv")

x_test = csv.read_csv(DATA_PATH + "x_test.csv")
y_test = np.genfromtxt(DATA_PATH + "y_test.csv")


#del X_train, Y_train

def objective(trial):
    """
    Intervalos de busqueda de parametros con modelo objetivo a optmizar
    """

    params = {
        'boosting': 'gbdt',
        "objective": "regression",
        "metric": "rmse",
        "force_row_wise": True,
        #"n_estimators": 1000, default
        "verbose": -1,
        #"bagging_freq": 1,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 100),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),

    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, Y_train)

    y_pred = model.predict(x_test)
    rmse = root_mean_squared_error(y_test, y_pred)

    return rmse


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)


with open(PROJECT_PATH +"outputs/best_parameters_lgbmv2.json", "w") as fp:
    json.dump(study.best_params , fp, indent=2)
