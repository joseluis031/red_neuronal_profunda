import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from keras.models import load_model



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

# Convertir fase, equipo_local y equipo_visitante en variables numéricas
le = LabelEncoder()
resultados['fase'] = le.fit_transform(resultados['fase'])
resultados['equipo_local'] = le.fit_transform(resultados['equipo_local'])
resultados['equipo_visitante'] = le.fit_transform(resultados['equipo_visitante'])

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = resultados[['fase', 'equipo_local', 'equipo_visitante']]
y = resultados[['goles_equipo_local', 'goles_equipo_visitante']]

# Normalizar características
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Cargar el modelo de red neuronal desde el archivo en modo binario
modelo = load_model('modelo_red_neuronal.h5')

# Leer eliminatoria actual
octavos = pd.read_csv("Eliminatoria actual/eliminatoria.csv")

X_octavos = octavos[['fase', 'equipo_local', 'equipo_visitante']]
X_octavos_encoded = pd.get_dummies(X_octavos)

# Normalizar características de la eliminatoria actual
X_octavos_normalized = scaler.transform(X_octavos_encoded)

# Realizar la predicción con el modelo de red neuronal
predicciones = modelo.predict(X_octavos_normalized)

# Rellenar valores faltantes con las predicciones
octavos['goles_equipo_local'] = np.round(predicciones[:, 0])
octavos['goles_equipo_visitante'] = np.round(predicciones[:, 1])

# Guardar datos actualizados en un nuevo archivo CSV
octavos.to_csv("resultado_octavos_red_neuronal.csv", index=False)
