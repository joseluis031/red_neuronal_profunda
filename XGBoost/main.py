
from Predecir_XGB.prediccion_XGB import *
from Manejo_datos_XGB.datos_XGB import *


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