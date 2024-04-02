"""
Decarga los datos del proyecto
y los descomprime en la carpeta data/raw_data
!! importante tener configurado el token de cuenta de kaggle
url de los datos: https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction
"""
import os
import zipfile


def download_data(raw_data_path):
    """
    Descarga y descomprime todos los archivos del proyecto

    Parameters: 
    --------
        raw_data_path(str): ruta de datos sin procesar
    """
    if os.path.exists(raw_data_path + "/new-york-city-taxi-fare-prediction.zip") is False:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files("new-york-city-taxi-fare-prediction",
            path=raw_data_path, force=False, quiet=False)

        # comprobar si esxisten los archivos csv
        # en caso de no exister decomprime el zip
        if  os.path.exists(raw_data_path + "/test.csv") is False or\
            os.path.exists(raw_data_path + "/train.csv") is False:

            with zipfile.ZipFile(raw_data_path + "/new-york-city-taxi-fare-prediction.zip", 'r') as zip_ref:
                zip_ref.extractall(raw_data_path)
        print("Descarga exitosa")
    else:
        print("Datos encontrados")



if __name__ == "__main__":
    PROJECT_PATH = os.path.dirname(__file__)

    try:
        download_data(PROJECT_PATH + "/data/raw_data")
    except Exception as e:
        print(e)
