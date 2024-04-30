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

# Filtrar los datos para incluir solo los partidos hasta la fase de cuartos de final
partidos_hasta_cuartos_df = partidos_df[partidos_df['fase'] != 'Cuartos']

# Función para simular un partido
def simular_partido(probabilidades):
    resultado = np.random.choice(['victoria', 'empate', 'derrota'], p=probabilidades)
    return resultado

# Simular múltiples temporadas
num_simulaciones = 1000
victorias_simuladas = {}

for _ in range(num_simulaciones):
    for fase in ['Cuartos', 'Semifinales', 'Final']:
        partidos_fase_actual = partidos_df[partidos_df['fase'] == fase]
        for _, partido in partidos_fase_actual.iterrows():
            probabilidades_local = [partido['prob_victoria_local'], partido['prob_empate'], partido['prob_derrota_local']]
            resultado_local = simular_partido(probabilidades_local)
            if resultado_local == 'victoria':
                if partido['equipo_local'] in victorias_simuladas:
                    victorias_simuladas[partido['equipo_local']] += 1
                else:
                    victorias_simuladas[partido['equipo_local']] = 1
            elif resultado_local == 'empate':
                continue
            else:
                probabilidades_visitante = [partido['prob_derrota_visitante'], partido['prob_empate'], partido['prob_victoria_visitante']]
                resultado_visitante = simular_partido(probabilidades_visitante)
                if resultado_visitante == 'victoria':
                    if partido['equipo_visitante'] in victorias_simuladas:
                        victorias_simuladas[partido['equipo_visitante']] += 1
                    else:
                        victorias_simuladas[partido['equipo_visitante']] = 1

# Identificar al equipo con más victorias simuladas
ganador_simulado = max(victorias_simuladas, key=victorias_simuladas.get)
victorias_totales_ganador = victorias_simuladas[ganador_simulado]

print("El ganador simulado de la Liga de Campeones es:", ganador_simulado)
print("Número total de victorias simuladas:", victorias_totales_ganador)
