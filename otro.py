import pandas as pd
import xgboost as xgb

# Cargar datos de entrenamiento
datos_entrenamiento = pd.read_csv("CSVS STATS/todas_stats.csv")

# Ajustar datos para tener una sola variable objetivo (goles totales en un partido)
datos_entrenamiento['Goles Totales'] = datos_entrenamiento['Goles Marcados Local_ult10_temp'] + datos_entrenamiento['Goles Marcados Visitante_ult10_temp']

# Separar características (X) y etiquetas (y)
X = datos_entrenamiento.drop(columns=['equipo', 'Goles Totales'])  # Eliminar columna del equipo y la nueva columna de goles totales
y = datos_entrenamiento['Goles Totales']  # Variable objetivo: goles totales en un partido

# Entrenar modelo XGBoost
modelo = xgb.XGBRegressor()
modelo.fit(X, y)

# Ahora, para predecir los goles en los enfrentamientos:
datos_a_predecir = pd.read_csv("Eliminatoria actual/cuartos.csv")

# Realizar predicciones
predicciones_enfrentamientos = modelo.predict(datos_a_predecir.drop(columns=['fase', 'equipo_local', 'equipo_visitante']))

# Agregar predicciones al DataFrame de los enfrentamientos
datos_a_predecir['goles_equipo_local'] = predicciones_enfrentamientos
datos_a_predecir['goles_equipo_visitante'] = predicciones_enfrentamientos

# Mostrar resultados de predicciones en enfrentamientos
print("Predicciones de goles en enfrentamientos:")
print(datos_a_predecir[['fase', 'equipo_local', 'goles_equipo_local', 'equipo_visitante', 'goles_equipo_visitante']])
