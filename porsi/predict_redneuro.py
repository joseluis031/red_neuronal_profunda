import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

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

# Dividir datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Definir modelo de red neuronal densa
modelo = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(2)
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Entrenar el modelo
modelo.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluar el modelo en el conjunto de prueba
loss, mae = modelo.evaluate(X_test, y_test)
print("Loss:", loss)
print("MAE:", mae)

# Guardar el modelo
modelo.save('modelo_red_neuronal.h5')
