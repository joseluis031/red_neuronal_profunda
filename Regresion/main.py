from Regresion.Manejo_Datos_REG.datos_REG import *
from Regresion.Predecir_REG.prediccion_Reg import *
from Regresion.Mediciones_REG.metricas_REG import *
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk




def main_reg():

    # Cargar datos de entrenamiento
    X, y = DataManager.cargar_datos("CSVS")

    # Dividir datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = DataManager.dividir_datos(X, y)

    # Crear instancia de LinearRegressor
    regresor = LinearRegressor()

    # Entrenar el modelo
    modelo = regresor.entrenar_modelo(X_train, y_train)

    # Cargar datos de la eliminatoria actual
    octavos = DataManager.cargar_eliminatoria("Eliminatoria actual/eliminatoria.csv")

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
    #DataManager.guardar_resultados(octavos, "resultado_final_regre.csv")
    
    
    
    
    
    # Calcular métricas de evaluación
    mse = Evaluator.calcular_mse(octavos[['goles_equipo_local', 'goles_equipo_visitante']], predicciones)
    r_squared = Evaluator.calcular_r2_score(octavos[['goles_equipo_local', 'goles_equipo_visitante']], predicciones)
    mae = Evaluator.calcular_mae(octavos[['goles_equipo_local', 'goles_equipo_visitante']], predicciones)
    
    print("MSE:", mse)
    print("R^2:", r_squared)
    print("MAE:", mae)
    
    
    # Mostrar el drawio.png 
    img = Image.open('Regresion/eliminatoria_regr.drawio.png')
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
    
    


