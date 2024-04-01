"""
Entremanieto de un modelo de LGBM, usa los parametros optimos obetenidos
del modulo parameter_tuninig.py, así es recomendable ejecutar ese script primero
sin embargo si no se encuentra un archivo json con parametros optimos el modelo se entrenara
con los valores por defecto
"""
import warnings
import os
import json
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error
import numpy as np
from pyarrow import csv
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")


def train_model(project_path, training_path):
    """
    Entrena modelo
    """
    #read_options = csv.ReadOptions(autogenerate_column_names=True)

    X_train = csv.read_csv(project_path + training_path + "/X_train.csv")
    Y_train = np.genfromtxt(project_path + training_path + "/Y_train.csv")

    x_test = csv.read_csv(project_path + training_path + "/x_test.csv")
    y_test = np.genfromtxt(project_path + training_path + "/y_test.csv")

    lgb_train = lgb.Dataset(X_train, Y_train,
                    feature_name=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
                    'dropoff_latitude', 'passenger_count',
                    'distance', 'pickup_time', 'pickup_year', 'pickup_month',
                    'pickup_hour', 'pickup_day'])
    lgb_eval = lgb.Dataset(x_test, y_test)

    # parametros iniciales
    params = {"boosting": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "force_row_wise": True,
            "verbose": -1}

    # incluir paramatros optimos
    try:
        f = open(project_path + '/outputs/best_parameters_lgbm.json')
        opt_params = json.load(f)
        params.update(opt_params)
    except Exception as e:
        print("No se encronto archivo con parametros")
        print("Se entranara el modelo con parametros por defecto")
        pass


    # ajustar modelo
    print("Entrenando modelo...")
    model = lgb.train(params,
                    train_set=lgb_train,
                    valid_sets=lgb_eval)


    model.save_model(project_path +"/model/lightgbm_run2.txt")
    print("Modelo guardado...")
    y_pred = model.predict(x_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"RMSE datos pruebas: {rmse:.2f}")


    lgb.plot_importance(model, title="Importancia de Variables",
                        importance_type="gain",
                        figsize=(15, 10), height=0.6,
                        precision=1,
                        color="#73c0de", grid =False)

    plt.savefig(project_path + '/outputs/feature_importance.png')


if __name__ == "__main__":
    PROJECT_PATH = os.path.dirname(__file__)
    DATA_PATH = "/data/to_train"

    train_model(PROJECT_PATH, DATA_PATH)
