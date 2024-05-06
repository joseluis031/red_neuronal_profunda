
from Predecir_XGB.prediccion_XGB import *
from Manejo_datos_XGB.datos_XGB import *
from Mediciones_XGB.metricas_XGB import *
from PIL import Image, ImageTk
import tkinter as tk

def main():
    predictor = PredictorXGBoost()
    data_manager = DataManager()
    
    # Cargar datos de entrenamiento
    X, y = data_manager.cargar_datos("CSVS")
    
    # Dividir datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = data_manager.dividir_datos(X, y)
    
    # Entrenar el modelo
    predictor.entrenar_modelo(X_train, y_train)
    
    # Realizar predicciones
    predicciones = predictor.predecir(X_test)
    
    # Calcular métricas utilizando la clase Evaluator
    mse = Evaluator.calcular_mse(y_test, predicciones)
    r_squared = Evaluator.calcular_r2_score(y_test, predicciones)
    mae = Evaluator.calcular_mae(y_test, predicciones)
    accuracy_score = Evaluator.calcular_accuracy(y_test, predicciones)
    
    print("MSE:", mse)
    print("R^2:", r_squared)
    print("MAE:", mae)
    print("Accuracy Score:", accuracy_score)
    
    # Cargar datos de la eliminatoria actual
    octavos = data_manager.cargar_eliminatoria("Eliminatoria actual/eliminatoria.csv")
    
    # Predecir resultados de los octavos de final
    X_octavos = octavos[['fase', 'equipo_local', 'equipo_visitante']]
    X_octavos_encoded = pd.get_dummies(X_octavos)
    
    X_test_octavos = pd.DataFrame(columns=X_train.columns, data=np.zeros((X_octavos_encoded.shape[0], X_train.shape[1])))
    for col in X_octavos_encoded.columns:
        if col in X_test_octavos.columns:
            X_test_octavos[col] = X_octavos_encoded[col]
    
    predicciones_octavos = predictor.predecir(X_test_octavos)
    
    # Rellenar valores faltantes con las predicciones y asegurarse de que sean enteros y positivos
    octavos[['goles_equipo_local', 'goles_equipo_visitante']] = np.round(np.maximum(predicciones_octavos, 0))
    
    
    
    
    # Guardar resultados en un nuevo archivo CSV
    #data_manager.guardar_resultados(octavos, "resultado_octavos_xgboost.csv")




    # Calcular métricas para las predicciones de los octavos de final
    mse_octavos = Evaluator.calcular_mse(octavos[['goles_equipo_local', 'goles_equipo_visitante']], predicciones_octavos)
    r_squared_octavos = Evaluator.calcular_r2_score(octavos[['goles_equipo_local', 'goles_equipo_visitante']], predicciones_octavos)
    mae_octavos = Evaluator.calcular_mae(octavos[['goles_equipo_local', 'goles_equipo_visitante']], predicciones_octavos)
    
    print("MSE para octavos de final:", mse_octavos)
    print("R^2 para octavos de final:", r_squared_octavos)
    print("MAE para octavos de final:", mae_octavos)
    
    
    
        # Mostrar el drawio.png 
    img = Image.open('XGBoost/eliminatoria_xgBoost.drawio.png')
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

