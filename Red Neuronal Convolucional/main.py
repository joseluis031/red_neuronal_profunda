from Escalado_CNN.scaler_CNN import *
from Manejo_Datos_CNN.datos_CNN import *
from Predecir_CNN.prediccion_CNN import *
from sklearn.model_selection import train_test_split
import numpy as np



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
    octavos.to_csv("Red Neuronal Convolucional/Resultados/resultado_final4_cnn.csv", index=False)

if __name__ == "__main__":
    main()
