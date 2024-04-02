"""
Procesado y particionado de los datos de la competencia
new-york-city-taxi-fare-prediction,

@author: Ismael Orihuela
"""
import os
import polars as pl


def read_taxi_data(file_path: str) -> pl.LazyFrame:
    """
    Lee el registro de viajes en uber en NY

    Parameters: 
    --------
        file_path(str): ubicacion del archivo csv
    """

    ldf = pl.scan_csv(file_path,
    separator=",",
    encoding="utf8",
    dtypes={"key": pl.String,
            "pickup_datetime": pl.String})

    return ldf


def drop_null_cordinates(ldf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Filtrar coordenas con valores nulo
    """

    ldf = ldf.drop_nulls(subset=["dropoff_longitude","dropoff_latitude"])
    return ldf


def filter_ny_zone(ldf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Filtrar viajes dentro de la zona de NY
    """
    ny_longitude = [-74.05, -73.69]
    ny_latitude = [40.55, 40.90]

    ldf = ldf.filter(
        pl.col("pickup_longitude") >= ny_longitude[0],
        pl.col("pickup_longitude") <= ny_longitude[1],
        pl.col("pickup_latitude") >= ny_latitude[0],
        pl.col("pickup_latitude") <= ny_latitude[1],
        # puntos de destino
        pl.col("dropoff_longitude") >= ny_longitude[0],
        pl.col("dropoff_longitude") <= ny_longitude[1],
        pl.col("dropoff_latitude") >= ny_latitude[0],
        pl.col("dropoff_latitude") <= ny_latitude[1]
        )
    return ldf


def filter_fare_amunt(ldf:pl.LazyFrame) -> pl.LazyFrame:
    """
    Filtrar viajes con tarifas menores a 100 dlls
    """
    ldf = ldf.filter(
        pl.col("fare_amount") > 0,
        pl.col("fare_amount") <= 100)
    return ldf


def calculate_harvesine_distance(ldf:pl.LazyFrame) -> pl.LazyFrame:
    """
    Calculo de la distancia de Harvesine entre puntos de partida y destino
    """

    ldf = ldf.with_columns(
    [
        pl.col("pickup_longitude").radians().alias("pickup_longitude_rad"),
        pl.col("pickup_latitude").radians().alias("pickup_latitude_rad"),
        pl.col("dropoff_longitude").radians().alias("dropoff_longitude_rad"),
        pl.col("dropoff_latitude").radians().alias("dropoff_latitude_rad")
    ]
    ).with_columns(
    [
        ((pl.col("dropoff_latitude_rad") - pl.col("pickup_latitude_rad"))/ 2).alias("diff_lat_2"),
        ((pl.col("dropoff_longitude_rad") - pl.col("pickup_longitude_rad"))/2).alias("diff_long_2"),
    ]
    ).with_columns(
          (pl.col("diff_lat_2").sin()**2 + (
          pl.col("dropoff_latitude_rad").cos() * pl.col("pickup_latitude_rad").cos()
          * pl.col("diff_long_2").sin()**2))
          .alias("inner_results")
    ).with_columns(
       (6371*2*pl.arctan2(
            pl.col("inner_results").sqrt(),
            (1-pl.col("inner_results")).sqrt()
        )).alias("distance")

    ).select(
        pl.exclude(["pickup_longitude_rad", "pickup_latitude_rad", "dropoff_longitude_rad",
                    "dropoff_latitude_rad", "diff_lat_2", "diff_long_2", "inner_results"]))

    return ldf


def filter_distance(ldf:pl.LazyFrame) -> pl.LazyFrame:
    """
    Filtrar viajes con distancia mayores a cero 
    """
    ldf = ldf.filter(pl.col("distance") > 0)
    return ldf


def create_date_columns(ldf:pl.LazyFrame) -> pl.LazyFrame:
    """
    Crear columnas con año, mes, dia y hora del viaje
    """
    ldf = ldf.with_columns(
        pl.col("pickup_datetime")
        .str.to_datetime(format="%Y-%m-%d %H:%M:%S %Z", strict=False)
        )

    ldf = ldf.with_columns(
        [
            pl.col("pickup_datetime").dt.hour().alias("pickup_time"),
            pl.col("pickup_datetime").dt.year().alias("pickup_year"),
            pl.col("pickup_datetime").dt.month().alias("pickup_month"),
            pl.col("pickup_datetime").dt.hour().alias("pickup_hour"),
            pl.col("pickup_datetime").dt.weekday().alias("pickup_day")
        ])

    return ldf


def calculate_fare_per_km(ldf:pl.LazyFrame) -> pl.LazyFrame:
    """
    Calcular la tarifa por km
    """
    ldf = ldf.with_columns(
    (pl.col("fare_amount") / pl.col("distance")).alias("fare_per_km"))
    return ldf


def filter_fare_per_km(ldf:pl.LazyFrame) -> pl.LazyFrame:
    """
    Filtrar viajes con una tarida por km menor a 50 KM
    """
    ldf = ldf.filter(pl.col("fare_per_km") < 50)
    return ldf


def def_select_columns(ldf:pl.LazyFrame, include_key:bool=False) -> pl.LazyFrame:
    """
    Selecciona solo las columnas a usar para el 
    entranmiento
    """
    if include_key:
        to_exclude = ["pickup_datetime"]
    else:
        to_exclude = ["key", "pickup_datetime"]

    ldf = ldf.select(pl.exclude(to_exclude))

    return ldf


def process_raw_data(ldf:pl.LazyFrame) -> pl.LazyFrame:
    """
    Aplicar todas la funciones de limpieza
    """
    ldf = ldf.pipe(filter_ny_zone)\
        .pipe(drop_null_cordinates)\
        .pipe(filter_fare_amunt)\
        .pipe(calculate_harvesine_distance)\
        .pipe(filter_distance)\
        .pipe(calculate_fare_per_km)\
        .pipe(filter_fare_per_km)\
        .pipe(create_date_columns)\
        .pipe(def_select_columns)

    return ldf

def process_test_data(ldf:pl.LazyFrame) -> pl.LazyFrame:
    """
    Aplicar todas la funciones de limpieza
    """
    ldf = ldf.pipe(drop_null_cordinates)\
        .pipe(calculate_harvesine_distance)\
        .pipe(create_date_columns)\
        .pipe(def_select_columns, include_key=True)

    return ldf


def split_train_test_data(file_to_split:str,
    output_folder:str, limit_rows:int=None, train_size: float=0.8):
    """
    Particiona los datos en dos conjuntos, entranamiento, pruebas y guarda los resultados 
    como csv

    Parameters: 
    ------
        file_to_split(str): ruta del archivo parquet a particionar
        limit_row(int): filtra solo las primera n filas
        train_size(float): Proporcion de datos dedicados al conjunto de entramiento

    Returns
    ------
        (x_train, y_train, x_test, y_tesst): conjuntos de entramiento y pruebas
    """
    print("Particionando datos...")
    ldf = pl.scan_parquet(file_to_split)

    if limit_rows:
        ldf = ldf.head(limit_rows)

    ldf = ldf.with_columns(pl.all().shuffle(seed=2024)).with_row_index()

    df_train = ldf.filter(pl.col("index") < pl.col("index").max() * train_size)
    df_test = ldf.filter(pl.col("index") >= pl.col("index").max() * train_size)

    df_train = df_train.select(pl.exclude("index"))
    df_test = df_test.select(pl.exclude("index"))

    # seleccionar predictores y objetivo
    X_df_train = df_train.select(pl.exclude(["fare_amount", "fare_per_km"]))
    Y_df_train = df_train.select(pl.col("fare_amount"))

    x_df_test = df_test.select(pl.exclude(["fare_amount", "fare_per_km"]))
    y_df_test = df_test.select(pl.col("fare_amount"))

    X_df_train.collect().write_csv(
        output_folder + "/X_train.csv", include_header=True)
    Y_df_train.collect().write_csv(
        output_folder + "/Y_train.csv", include_header=False)

    x_df_test.collect().write_csv(
        output_folder + "/x_test.csv", include_header=True)
    y_df_test.collect().write_csv(
        output_folder + "/y_test.csv", include_header=False)


def export_data(raw_data_path:str, output_path:str,
                mode:str = "over_write", output:str=None):
    """
    Lee todos los registros, aplica las transformaciones y exporta los datos como parquet
    
    Parameters:
    -----------
        raw_data_path: Ruta del archivo csv sin procesar
        mode(str): {over_write, preserve} si el valor es over_write sobre escribe el archivo 
            con datos limpios
            si el valor es preserve, y existe un archivo con los datos limpios omite el proceso
    """

    file_path =  f"{output_path}/{output}.parquet"
    clean_data_exits = os.path.exists(file_path)

    if (mode =="over_write")\
        or (mode=="preserve" and clean_data_exits is False):

        taxi_data = read_taxi_data(raw_data_path)

        print("Procesando datos...")
        if output == "train":
            taxi_data = process_raw_data(taxi_data)
        else:
            taxi_data = process_test_data(taxi_data)

        print("Guardando datos...")
        taxi_data.collect().write_parquet(file_path)

    print(f"limpieza de datos {output}: finalizado")



if __name__ == "__main__":

    project_path = os.path.dirname(__file__)
    # leer archivo y aplicar tranformaciones

    # y exportar como parquet
    export_data(project_path + "/data/raw_data/train.csv", "preserve",
                output="train")
    export_data(project_path + "/data/raw_data/test.csv", "preserve",
                output="test")

    # split data
    print("particionando datos")
    split_train_test_data(project_path + "/data/clean/train.parquet",
        output_folder=project_path + "/data/to_train",
        limit_rows=100_0000, train_size=0.8)

    print("Finalizado")
