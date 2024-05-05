import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
    
    def cargar_datos_eliminacion(self):
        return pd.read_csv(self.archivo_eliminacion)

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=50)
    
    def entrenar(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predecir(self, X_prediccion):
        return self.model.predict(X_prediccion)

def main():
    data_handler = DataHandler("CSVS", "Eliminatoria actual/final.csv")
    
    # Cargar datos
    resultados = data_handler.cargar_datos_entrenamiento()
    octavos = data_handler.cargar_datos_eliminacion()
    
    # Dividir el conjunto de datos en características (X) y etiquetas (y)
    X = resultados[['fase', 'equipo_local', 'equipo_visitante']]
    y = resultados[['goles_equipo_local', 'goles_equipo_visitante']]
    
    # Codificar características categóricas
    X = pd.get_dummies(X)
    
    # Dividir datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    
    # Entrenar el modelo de Random Forest
    rf_model = RandomForestModel()
    rf_model.entrenar(X_train, y_train)
    
    # Preparar datos de predicción
    X_octavos = octavos[['fase', 'equipo_local', 'equipo_visitante']]
    X_octavos_encoded = pd.get_dummies(X_octavos)
    
    # Asegurarse de que los datos de predicción tengan todas las características presentes en los datos de entrenamiento
    X_octavos_encoded = X_octavos_encoded.reindex(columns=X.columns, fill_value=0)
    
    # Realizar predicciones
    predicciones = rf_model.predecir(X_octavos_encoded)
    
    # Rellenar valores faltantes con las predicciones y asegurarse de que sean enteros y positivos
    octavos[['goles_equipo_local', 'goles_equipo_visitante']] = np.round(predicciones)
    
    # Guardar datos actualizados en un nuevo archivo CSV
    octavos.to_csv("resultado_octavos.csv", index=False)

if __name__ == "__main__":
    main()
