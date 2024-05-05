import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np

class PredictorXGBoost:
    def __init__(self):
        self.modelo = XGBRegressor()
    
    def entrenar_modelo(self, X_train, y_train):
        self.modelo.fit(X_train, y_train)
    
    def predecir(self, X_prediccion):
        return self.modelo.predict(X_prediccion)

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

def main():
    predictor = PredictorXGBoost()
    data_manager = DataManager()
    
    # Cargar datos de entrenamiento
    X, y = data_manager.cargar_datos("CSVS")
    
    # Dividir datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = data_manager.dividir_datos(X, y)
    
    # Entrenar el modelo
    predictor.entrenar_modelo(X_train, y_train)
    
    # Cargar datos de la eliminatoria actual
    octavos = data_manager.cargar_eliminatoria("Eliminatoria actual/eliminatoria.csv")
    
    # Predecir resultados de los octavos de final
    X_octavos = octavos[['fase', 'equipo_local', 'equipo_visitante']]
    X_octavos_encoded = pd.get_dummies(X_octavos)
    
    X_test_octavos = pd.DataFrame(columns=X_train.columns, data=np.zeros((X_octavos_encoded.shape[0], X_train.shape[1])))
    for col in X_octavos_encoded.columns:
        if col in X_test_octavos.columns:
            X_test_octavos[col] = X_octavos_encoded[col]
    
    predicciones = predictor.predecir(X_test_octavos)
    
    # Rellenar valores faltantes con las predicciones y asegurarse de que sean enteros y positivos
    octavos[['goles_equipo_local', 'goles_equipo_visitante']] = np.round(np.maximum(predicciones, 0))
    
    # Guardar resultados en un nuevo archivo CSV
    data_manager.guardar_resultados(octavos, "resultado_octavos_xgboost.csv")

if __name__ == "__main__":
    main()
