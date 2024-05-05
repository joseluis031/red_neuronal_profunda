import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

class DataHandler:
    def __init__(self, carpeta_entrenamiento, archivo_prediccion):
        self.carpeta_entrenamiento = carpeta_entrenamiento
        self.archivo_prediccion = archivo_prediccion
    
    def cargar_datos_entrenamiento(self):
        archivos_csv = [os.path.join(self.carpeta_entrenamiento, archivo) for archivo in os.listdir(self.carpeta_entrenamiento) if archivo.startswith("temp")]
        dataframes = []
        for archivo in archivos_csv:
            df = pd.read_csv(archivo)
            dataframes.append(df)
        resultados = pd.concat(dataframes, ignore_index=True)
        resultados = resultados.dropna(subset=['goles_equipo_local', 'goles_equipo_visitante'])
        return resultados
    
    def cargar_datos_prediccion(self):
        return pd.read_csv(self.archivo_prediccion)

class Scaler:
    def __init__(self):
        self.scaler = None
    
    def fit_transform(self, X_train):
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X_train)
    
    def transform(self, X_test):
        return self.scaler.transform(X_test)

class ConvolutionalNeuralNetwork:
    def __init__(self, input_shape):
        self.model = Sequential([
            Conv1D(64, 2, activation='relu', input_shape=input_shape),
            MaxPooling1D(1),
            Conv1D(64, 2, activation='relu'),
            MaxPooling1D(1),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(2)  # 2 neuronas de salida para predecir goles_equipo_local y goles_equipo_visitante
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
    
    def entrenar(self, X_train, y_train):
        self.history = self.model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    
    def evaluar(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predecir(self, X_prediccion):
        return self.model.predict(X_prediccion)

def main():
    data_handler = DataHandler("CSVS", "Eliminatoria actual/eliminatoria.csv")
    
    # Cargar datos
    resultados = data_handler.cargar_datos_entrenamiento()
    octavos = data_handler.cargar_datos_prediccion()
    
    # Obtener una lista de todos los equipos presentes en los datos de entrenamiento y los datos de predicción
    equipos_entrenamiento = set(resultados['equipo_local']).union(set(resultados['equipo_visitante']))
    equipos_octavos = set(octavos['equipo_local']).union(set(octavos['equipo_visitante']))
    
    # Filtrar los equipos presentes en los datos de entrenamiento que también están en los datos de predicción
    equipos_comunes = list(equipos_entrenamiento.intersection(equipos_octavos))
    
    # Filtrar los datos de entrenamiento para incluir solo los partidos que involucran equipos comunes
    resultados_filtrados = resultados[(resultados['equipo_local'].isin(equipos_comunes)) & (resultados['equipo_visitante'].isin(equipos_comunes))]
    
    # Dividir el conjunto de datos en características (X) y etiquetas (y)
    X = resultados_filtrados[['fase', 'equipo_local', 'equipo_visitante']]
    y = resultados_filtrados[['goles_equipo_local', 'goles_equipo_visitante']]
    
    # Codificar características categóricas
    X = pd.get_dummies(X)
    
    # Dividir datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
    
    # Crear el escalador y escalar características
    scaler = Scaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Redimensionar datos para que se ajusten a la entrada de Conv1D
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    # Crear modelo de red neuronal convolucional
    cnn_model = ConvolutionalNeuralNetwork(input_shape=X_train_reshaped.shape[1:])
    
    # Entrenar el modelo
    cnn_model.entrenar(X_train_reshaped, y_train)
    
    # Evaluar el modelo en el conjunto de prueba
    loss = cnn_model.evaluar(X_test_reshaped, y_test)
    print("Loss en conjunto de prueba:", loss)
    
    # Preparar datos de predicción
    X_octavos = octavos[(octavos['equipo_local'].isin(equipos_comunes)) & (octavos['equipo_visitante'].isin(equipos_comunes))]
    X_octavos_encoded = pd.get_dummies(X_octavos)
    
    # Asegurarse de que los datos de predicción tengan todas las características presentes en los datos de entrenamiento
    columnas_entrenamiento = X.columns
    for columna in columnas_entrenamiento:
        if columna not in X_octavos_encoded.columns:
            X_octavos_encoded[columna] = 0
    
    # Ordenar las columnas para que estén en el mismo orden que en los datos de entrenamiento
    X_octavos_encoded = X_octavos_encoded[columnas_entrenamiento]
    
    # Escalar características de predicción
    X_octavos_scaled = scaler.transform(X_octavos_encoded)
    X_octavos_reshaped = X_octavos_scaled.reshape(X_octavos_scaled.shape[0], X_octavos_scaled.shape[1], 1)
    
    # Realizar predicciones
    predicciones = cnn_model.predecir(X_octavos_reshaped)
    
    # Rellenar valores faltantes con las predicciones y asegurarse de que sean enteros y positivos
    octavos['goles_equipo_local'] = np.round(np.maximum(predicciones[:, 0], 0))
    octavos['goles_equipo_visitante'] = np.round(np.maximum(predicciones[:, 1], 0))
    
    # Guardar datos actualizados en un nuevo archivo CSV
    octavos.to_csv("Red Neuronal Convolucional/Resultados/resultado_final_cnn.csv", index=False)

if __name__ == "__main__":
    main()
