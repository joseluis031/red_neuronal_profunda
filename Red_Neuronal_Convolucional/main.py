from Red_Neuronal_Convolucional.Escalado_CNN.scaler_CNN import *
from Red_Neuronal_Convolucional.Manejo_Datos_CNN.datos_CNN import *
from Red_Neuronal_Convolucional.Predecir_CNN.prediccion_CNN import *
from sklearn.model_selection import train_test_split
import numpy as np
from Red_Neuronal_Convolucional.Mediciones_CNN.metricas_CNN import *
from PIL import Image, ImageTk
import tkinter as tk



def main_CNN():
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
    #octavos.to_csv("Red Neuronal Convolucional/Resultados/resultado_final4_cnn.csv", index=False)
    
    
    
    
    # Calcular métricas
    mse = Evaluator.calcular_mse(octavos[['goles_equipo_local', 'goles_equipo_visitante']], predicciones)
    r_squared = Evaluator.calcular_r2_score(octavos[['goles_equipo_local', 'goles_equipo_visitante']], predicciones)
    mae = Evaluator.calcular_mae(octavos[['goles_equipo_local', 'goles_equipo_visitante']], predicciones)

    print("MSE:", mse)
    print("R^2:", r_squared)
    print("MAE:", mae)

    # Mostrar el drawio.png 
    img = Image.open('Red_neuronal_Convolucional/eliminatoria_red_neuronal_conv.drawio.png')
    # Crear una ventana Tkinter
    root = tk.Tk()
    root.title("Imagen")

    # Convertir la imagen para Tkinter
    img_tk = ImageTk.PhotoImage(img)

    # Mostrar la imagen en un widget Label
    label_img = tk.Label(root, image=img_tk)
    label_img.pack()

    # Centrar la ventana en la pantalla
    root.eval('tk::PlaceWindow . center')

    # Ejecutar el bucle principal de Tkinter
    root.mainloop()
