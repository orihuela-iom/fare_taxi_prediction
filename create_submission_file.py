import os
import polars as pl
import lightgbm as lgb


def creat_file_to_submission(projec_path:str):
    """
    Carga un modelo de lightgbm, hace predicciones sobre los datos de test del proyecto
    y guarda los resultado como un archivo csv, con las colulmas {'key', 'fare_amount'}
    
    Parameters:
    ---------
        projec_path(str): folder principal del proyect
    """

    submission_data = pl.read_parquet(projec_path + "/data/clean/test.parquet")

    to_test = submission_data.select(pl.exclude("key"))
    to_test = to_test.to_numpy()

    # cargar modelo

    model = lgb.Booster(
        model_file=projec_path + "/model/lightgbm_best.txt")


    y_pred = model.predict(to_test)

    to_submission = submission_data.select(pl.col("key"))

    to_submission = to_submission.with_columns(
        fare_amount=y_pred)

    to_submission.write_csv(projec_path + "/data/submission.csv")
    print(f"Archivo creado en: {projec_path + '/data/submission.csv'}")


if __name__ == "__main__":

    PROJECT_PATH = os.path.dirname(__file__)
    creat_file_to_submission(PROJECT_PATH)

