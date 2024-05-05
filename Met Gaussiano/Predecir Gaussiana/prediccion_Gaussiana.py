import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np

class DataHandler:
    def __init__(self, carpeta, archivo_eliminacion):
        self.carpeta = carpeta
        self.archivo_eliminacion = archivo_eliminacion
    
    def cargar_datos_entrenamiento(self):
        archivos_csv = [os.path.join(self.carpeta, archivo) for archivo in os.listdir(self.carpeta) if archivo.startswith("temp")]
        dataframes = [pd.read_csv(archivo) for archivo in archivos_csv]
        resultados = pd.concat(dataframes, ignore_index=True)
        resultados = resultados.dropna(subset=['goles_equipo_local', 'goles_equipo_visitante'])
        return resultados
    
    def cargar_datos_eliminacion(self, archivo):
        return pd.read_csv(archivo)

class GaussianProcessModel:
    def __init__(self, random_state=0, length_scale=2.0):
        self.random_state = random_state
        self.length_scale = length_scale
    
    def entrenar(self, X_train, y_train):
        kernel = 0.5 * RBF(length_scale=self.length_scale)
        self.modelo = GaussianProcessRegressor(kernel=kernel, random_state=self.random_state)
        self.modelo.fit(X_train, y_train)
    
    def predecir(self, X_prediccion):
        return self.modelo.predict(X_prediccion)

def main():
    data_handler = DataHandler("CSVS", "Eliminatoria actual/final.csv")
    
    # Cargar datos
    resultados = data_handler.cargar_datos_entrenamiento()
    octavos = data_handler.cargar_datos_eliminacion("Eliminatoria actual/eliminatoria.csv")
    
    # Dividir el conjunto de datos en características (X) y etiquetas (y)
    X = resultados[['fase', 'equipo_local', 'equipo_visitante']]
    y = resultados[['goles_equipo_local', 'goles_equipo_visitante']]
    
    # Codificar características categóricas
    X = pd.get_dummies(X)
    
    # Dividir datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    
    # Entrenar el modelo de regresión gaussiana
    gaussian_model = GaussianProcessModel()
    gaussian_model.entrenar(X_train, y_train)
    
    # Preparar datos de predicción
    X_octavos = octavos[['fase', 'equipo_local', 'equipo_visitante']]
    X_octavos_encoded = pd.get_dummies(X_octavos)
    
    # Alinear las columnas de los datos de predicción con las del entrenamiento
    columnas_entrenamiento = X_train.columns
    X_octavos_encoded_alineado = X_octavos_encoded.reindex(columns=columnas_entrenamiento, fill_value=0)
    
    # Realizar predicciones para los octavos de final
    predicciones = gaussian_model.predecir(X_octavos_encoded_alineado)
    
    # Rellenar valores faltantes con las predicciones y asegurarse de que sean enteros y positivos
    octavos[['goles_equipo_local', 'goles_equipo_visitante']] = np.round(np.maximum(predicciones, 0))
    
    # Guardar datos actualizados en un nuevo archivo CSV
    octavos.to_csv("resultado_final_gaussiano.csv", index=False)

if __name__ == "__main__":
    main()
