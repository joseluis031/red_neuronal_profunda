import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np

# Obtener la lista de archivos CSV en la misma carpeta
carpeta = "CSVS"
archivos_csv = [os.path.join(carpeta, archivo) for archivo in os.listdir(carpeta) if archivo.startswith("temp")]

# Lista para almacenar cada DataFrame
dataframes = []

# Leer cada archivo CSV y almacenar su contenido en la lista de DataFrames
for archivo in archivos_csv:
    df = pd.read_csv(archivo)
    dataframes.append(df)

resultados = pd.concat(dataframes, ignore_index=True)

# Eliminar filas con valores faltantes en goles_local y goles_visitante
resultados = resultados.dropna(subset=['goles_equipo_local', 'goles_equipo_visitante'])

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = resultados[['fase', 'equipo_local', 'equipo_visitante']]
y = resultados[['goles_equipo_local', 'goles_equipo_visitante']]

X = pd.get_dummies(X)

# Dividir datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Entrenar el modelo de regresión gaussiana
kernel = 0.5 * RBF(length_scale=1.0)
modelo = GaussianProcessRegressor(kernel=kernel, random_state=0)
modelo.fit(X_train, y_train)

# Leer eliminatoria actual
octavos = pd.read_csv("Eliminatoria actual/eliminatoria.csv")

X_octavos = octavos[['fase', 'equipo_local', 'equipo_visitante']]
X_octavos_encoded = pd.get_dummies(X_octavos)

# Obtener las columnas de los datos de entrenamiento
columnas_entrenamiento = X_train.columns

# Alinear las columnas de los datos de predicción con las del entrenamiento
X_octavos_encoded_alineado = X_octavos_encoded.reindex(columns=columnas_entrenamiento, fill_value=0)

# Realizar predicciones para los octavos de final
predicciones = modelo.predict(X_octavos_encoded_alineado)

# Rellenar valores faltantes con las predicciones y asegurarse de que sean enteros y positivos
octavos[['goles_equipo_local', 'goles_equipo_visitante']] = np.round(np.maximum(predicciones, 0))

# Guardar datos actualizados en un nuevo archivo CSV
octavos.to_csv("resultado_octavos2_gaussiano.csv", index=False)
