import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Paso 1: Preparación de los datos
datos_entrenamiento = pd.read_csv("CSVS/temp2023_24.csv")
datos_prediccion = pd.read_csv("Eliminatoria actual/cuartos.csv")

# Preprocesamiento de datos (omitiendo la parte específica del preprocesamiento)

# Separar características (X) y etiquetas (y)
X_train = datos_entrenamiento.drop(columns=['equipo'])
y_train = datos_entrenamiento[['goles_equipo_local', 'goles_equipo_visitante']]

# Paso 2: Construcción del modelo
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(2))  # Dos salidas: goles_equipo_local y goles_equipo_visitante

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Paso 3: Entrenamiento del modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Paso 4: Predicción
X_prediccion = datos_prediccion.drop(columns=['fase', 'equipo_local', 'equipo_visitante'])
predicciones = model.predict(X_prediccion)

# Agregar las predicciones al DataFrame de predicción
datos_prediccion['goles_prediccion_equipo_local'] = predicciones[:, 0]
datos_prediccion['goles_prediccion_equipo_visitante'] = predicciones[:, 1]

# Imprimir los resultados de predicción
print(datos_prediccion[['fase', 'equipo_local', 'goles_prediccion_equipo_local', 'equipo_visitante', 'goles_prediccion_equipo_visitante']])
