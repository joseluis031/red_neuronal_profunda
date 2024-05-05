import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
import numpy as np

# Obtener la lista de archivos CSV en la carpeta de entrenamiento
carpeta_entrenamiento = "CSVS"
archivos_csv_entrenamiento = [os.path.join(carpeta_entrenamiento, archivo) for archivo in os.listdir(carpeta_entrenamiento) if archivo.startswith("temp")]

# Lista para almacenar cada DataFrame de entrenamiento
dataframes_entrenamiento = []

# Leer cada archivo CSV de entrenamiento y almacenar su contenido en la lista de DataFrames
for archivo in archivos_csv_entrenamiento:
    df_entrenamiento = pd.read_csv(archivo)
    dataframes_entrenamiento.append(df_entrenamiento)

# Concatenar todos los DataFrames de entrenamiento en uno solo
datos_entrenamiento = pd.concat(dataframes_entrenamiento, ignore_index=True)

# Dividir el conjunto de datos de entrenamiento en características (X) y etiquetas (y)
X_entrenamiento = datos_entrenamiento[['equipo_local', 'equipo_visitante']]
y_entrenamiento = datos_entrenamiento[['goles_equipo_local', 'goles_equipo_visitante']]

# Normalizar características de entrenamiento
scaler = StandardScaler()
X_entrenamiento = scaler.fit_transform(X_entrenamiento)

# Entrenar el modelo de regresión gaussiana
modelo_gaussiano = KernelRidge(kernel='rbf')
modelo_gaussiano.fit(X_entrenamiento, y_entrenamiento)

# Leer el CSV de predicciones
archivo_prediccion = "Eliminatoria actual/eliminatoria.csv"
datos_prediccion = pd.read_csv(archivo_prediccion)

# Normalizar características de predicción
X_prediccion = datos_prediccion[['equipo_local', 'equipo_visitante']]
X_prediccion = scaler.transform(X_prediccion)

# Realizar la predicción con el modelo gaussiano
predicciones = modelo_gaussiano.predict(X_prediccion)

# Rellenar valores faltantes con las predicciones
datos_prediccion['goles_equipo_local'] = np.round(predicciones[:, 0])
datos_prediccion['goles_equipo_visitante'] = np.round(predicciones[:, 1])

# Guardar datos actualizados en un nuevo archivo CSV
archivo_resultado = "ruta/del/archivo/de/resultado.csv"
datos_prediccion.to_csv(archivo_resultado, index=False)
