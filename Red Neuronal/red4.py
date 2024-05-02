import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# Leer los datos de entrenamiento
carpeta = "CSVS"
archivos_csv = [os.path.join(carpeta, archivo) for archivo in os.listdir(carpeta) if archivo.startswith("temp")]

dataframes = []
for archivo in archivos_csv:
    df = pd.read_csv(archivo)
    dataframes.append(df)

resultados = pd.concat(dataframes, ignore_index=True)
resultados = resultados.dropna(subset=['goles_equipo_local', 'goles_equipo_visitante'])

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = resultados[['fase', 'equipo_local', 'equipo_visitante']]
y = resultados[['goles_equipo_local', 'goles_equipo_visitante']]

# Codificar características categóricas
X = pd.get_dummies(X)

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar el modelo de red neuronal (código previo)

# Cargar el modelo entrenado
modelo = load_model('modelo_red_neuronal.h5')

# Leer los datos de predicción
octavos = pd.read_csv("Eliminatoria actual/final.csv")

# Filtrar los equipos de los datos de predicción para incluir solo los equipos presentes en los datos de entrenamiento
equipos_entrenamiento = ['Inter', 'Lyon', 'Chelsea', 'RB Salzburgo', 'Nápoles', 'Ajax', 'Benfica', 'Dortmund']
octavos_filtrados = octavos[(octavos['equipo_local'].isin(equipos_entrenamiento)) & (octavos['equipo_visitante'].isin(equipos_entrenamiento))]

# Preparar datos de predicción
X_octavos = octavos_filtrados[['fase', 'equipo_local', 'equipo_visitante']]
X_octavos_encoded = pd.get_dummies(X_octavos)
X_octavos_scaled = scaler.transform(X_octavos_encoded)

# Realizar predicciones
predicciones = modelo.predict(X_octavos_scaled)

# Rellenar valores faltantes con las predicciones y asegurarse de que sean enteros y positivos
octavos_filtrados[['goles_equipo_local', 'goles_equipo_visitante']] = np.round(np.maximum(predicciones, 0))

# Guardar datos actualizados en un nuevo archivo CSV
octavos_filtrados.to_csv("resultado_final_red_neuronal.csv", index=False)
