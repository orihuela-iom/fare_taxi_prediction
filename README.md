# fare_taxi_prediction
Pronostico tarifa taxi 


# Instalar requerimientos


```
pip install -r requirements.txt
```

Para descargar los datos de kaggle se requiere tener configurado el api token

# Estructura de proyecto

```
├── data/
│   ├── raw_data/
│   │   └── *.csv
│   ├── to_train/
│   │    └── *.csv
│   └── clean/
│       └── *.parquet
├── model
│   └── lightgbm_best.txt
├── outputs/
│    └── *
│
├── EDA.ipynb # análisis exploratorio
├── clean_data.py # limpieza de datos
├── parameter_tunning.py # optimización de parametros
├── train.py #entrenar modelo lgbm
├── create_submission_file.py # crear predicciones
└── run.py s# aarchivo principal

```

## Ejecutar todo el proyecto
Ejecuta el pipeline completo de datos

Descargar datos -> limpiar datos -> 
    Entrenar modelo -> Predicciones sobre datos test

```
python run.py
```

