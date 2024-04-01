"""
Ejecuta el proyecto completo
"""
import os
from get_data import download_data
from clean_data import export_data, split_train_test_data
from train import train_model
from create_submission_file import creat_file_to_submission


PROJECT_PATH = os.path.dirname(__file__)

# ruta de archivos a usar
RAW_DATA_PATH = "/data/raw_data"
CLEAN_DATA_PATH = "/data/clean"
TRAINING_DATA_PATH = "/data/to_train"

# split datos para entrenar
TRAIN_DATA_SIZE = 10_000_000
TRAINING_PROPORTION = 0.8


# 01 Descargar datos
download_data(PROJECT_PATH + RAW_DATA_PATH)

# 03 Descargar datos
# limpiar datos train
export_data(raw_data_path=PROJECT_PATH + RAW_DATA_PATH + "/train.csv",
            output_path=PROJECT_PATH + CLEAN_DATA_PATH,
            mode="preserve",
            output="train")

# limpiar datos test
export_data(raw_data_path=PROJECT_PATH + RAW_DATA_PATH + "/test.csv",
            output_path=PROJECT_PATH + CLEAN_DATA_PATH,
            mode="preserve",
            output="test")

# 04 Particionar datos en pruebas y test para entrenamiento
split_train_test_data(
    file_to_split= PROJECT_PATH + CLEAN_DATA_PATH + "/train.parquet",
    output_folder=PROJECT_PATH + TRAINING_DATA_PATH,
    limit_rows=TRAIN_DATA_SIZE, train_size=TRAINING_PROPORTION)

# 05 entrenar modelo
train_model(PROJECT_PATH, TRAINING_DATA_PATH)

# crear arhivo para competencia
creat_file_to_submission(PROJECT_PATH)
