import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataManager:
    @staticmethod
    def cargar_datos(carpeta):
        archivos_csv = [os.path.join(carpeta, archivo) for archivo in os.listdir(carpeta) if archivo.startswith("temp")]
        dataframes = []

        for archivo in archivos_csv:
            df = pd.read_csv(archivo)
            dataframes.append(df)

        resultados = pd.concat(dataframes, ignore_index=True)
        resultados = resultados.dropna(subset=['goles_equipo_local', 'goles_equipo_visitante'])

        X = resultados[['fase', 'equipo_local', 'equipo_visitante']]
        y = resultados[['goles_equipo_local', 'goles_equipo_visitante']]

        X = pd.get_dummies(X)

        return X, y

    @staticmethod
    def dividir_datos(X, y, test_size=0.2, random_state=50):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    @staticmethod
    def cargar_eliminatoria(ruta_archivo):
        return pd.read_csv(ruta_archivo)

    @staticmethod
    def guardar_resultados(resultado, ruta_archivo):
        resultado.to_csv(ruta_archivo, index=False)
