import os
import pandas as pd
import numpy as np

# Función para cargar todos los archivos CSV de una carpeta
def cargar_datos_desde_carpeta(ruta_carpeta):
    archivos_csv = [archivo for archivo in os.listdir(ruta_carpeta) if archivo.startswith('temp')]
    dataframes = []
    for archivo in archivos_csv:
        ruta_completa = os.path.join(ruta_carpeta, archivo)
        df = pd.read_csv(ruta_completa)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Cargar los datos de partidos desde la carpeta
ruta_carpeta_partidos = 'CSVS'
partidos_df = cargar_datos_desde_carpeta(ruta_carpeta_partidos)



# Filtrar solo los partidos del Real Madrid
partidos_real_madrid_df = partidos_df[(partidos_df['equipo_local'] == 'Real Madrid') | (partidos_df['equipo_visitante'] == 'Real Madrid')]

# Función para simular un partido
def simular_partido():
    # Supongamos que el Real Madrid tiene un 80% de probabilidad de ganar, 10% de empatar y 10% de perder un partido
    resultado = np.random.choice(['victoria', 'empate', 'derrota'], p=[0.8, 0.1, 0.1])
    return resultado

# Simular múltiples temporadas
num_simulaciones = 50
victorias_simuladas = []
empates_simulados = []
derrotas_simuladas = []

for _ in range(num_simulaciones):
    victorias_temporada = 0
    empates_temporada = 0
    derrotas_temporada = 0
    for _, partido in partidos_real_madrid_df.iterrows():
        resultado = simular_partido()
        if resultado == 'victoria' and (partido['equipo_local'] == 'Real Madrid' or partido['equipo_visitante'] == 'Real Madrid'):
            victorias_temporada += 1
        elif resultado == 'empate' and ('Real Madrid' in [partido['equipo_local'], partido['equipo_visitante']]):
            empates_temporada += 1
        elif resultado == 'derrota' and ('Real Madrid' in [partido['equipo_local'], partido['equipo_visitante']]):
            derrotas_temporada += 1
    victorias_simuladas.append(victorias_temporada)
    empates_simulados.append(empates_temporada)
    derrotas_simuladas.append(derrotas_temporada)

# Calcular estadísticas de las simulaciones
promedio_victorias = round(np.mean(victorias_simuladas))
desviacion_estandar_victorias = (np.std(victorias_simuladas))

promedio_empates = round(np.mean(empates_simulados))
desviacion_estandar_empates = (np.std(empates_simulados))

promedio_derrotas = round(np.mean(derrotas_simuladas))
desviacion_estandar_derrotas = (np.std(derrotas_simuladas))



print("Promedio de victorias:", promedio_victorias)
print("Desviación estándar de victorias:", desviacion_estandar_victorias)

print("Promedio de empates:", promedio_empates)
print("Desviación estándar de empates:", desviacion_estandar_empates)

print("Promedio de derrotas:", promedio_derrotas)
print("Desviación estándar de derrotas:", desviacion_estandar_derrotas)
