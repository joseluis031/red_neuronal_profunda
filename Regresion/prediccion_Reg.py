import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

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

class LinearRegressor:
    def __init__(self):
        self.modelo = LinearRegression()

    @staticmethod
    def entrenar_modelo(X_train, y_train):
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        return modelo

    @staticmethod
    def predecir(modelo, X_prediccion):
        return modelo.predict(X_prediccion)

def main():
    # Cargar datos de entrenamiento
    X, y = DataManager.cargar_datos("CSVS")

    # Dividir datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = DataManager.dividir_datos(X, y)

    # Crear instancia de LinearRegressor
    regresor = LinearRegressor()

    # Entrenar el modelo
    modelo = regresor.entrenar_modelo(X_train, y_train)

    # Cargar datos de la eliminatoria actual
    octavos = DataManager.cargar_eliminatoria("Eliminatoria actual/final.csv")

    X_octavos = octavos[['fase', 'equipo_local', 'equipo_visitante']]
    X_octavos_encoded = pd.get_dummies(X_octavos)

    X_test_octavos = pd.DataFrame(columns=X_train.columns, data=np.zeros((X_octavos_encoded.shape[0], X_train.shape[1])))
    for col in X_octavos_encoded.columns:
        if col in X_test_octavos.columns:
            X_test_octavos[col] = X_octavos_encoded[col]

    # Predecir resultados de la eliminatoria
    predicciones = LinearRegressor.predecir(modelo, X_test_octavos)

    # Rellenar valores faltantes con las predicciones y asegurarse de que sean enteros y positivos
    octavos[['goles_equipo_local', 'goles_equipo_visitante']] = np.round(np.maximum(predicciones, 0))

    # Guardar resultados en un nuevo archivo CSV
    DataManager.guardar_resultados(octavos, "resultado_final_regre.csv")

if __name__ == "__main__":
    main()
