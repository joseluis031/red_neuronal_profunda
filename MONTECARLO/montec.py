import pandas as pd
import numpy as np
import os

def cargar_datos_desde_carpeta(ruta_carpeta):
    archivos_csv = [archivo for archivo in os.listdir(ruta_carpeta) if archivo.startswith('temp')]
    dataframes = []
    for archivo in archivos_csv:
        ruta_completa = os.path.join(ruta_carpeta, archivo)
        df = pd.read_csv(ruta_completa)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def simular_partido():
    goles_local = np.random.poisson(1.8) # Promedio de goles en casa
    goles_visitante = np.random.poisson(1.2) # Promedio de goles de visitante
    if goles_local > goles_visitante:
        return 'local'
    elif goles_local < goles_visitante:
        return 'visitante'
    else:
        return 'empate'

def main():
    # Cargar los datos de partidos desde la carpeta
    ruta_carpeta_partidos = 'CSVS'
    partidos_df = cargar_datos_desde_carpeta(ruta_carpeta_partidos)

    # Leer el archivo CSV con los equipos clasificados
    ruta_archivo_equipos_clasificados = 'Eliminatoria actual/eliminatoria.csv'
    equipos_clasificados_df = pd.read_csv(ruta_archivo_equipos_clasificados)

    # Equipos clasificados
    equipos_clasificados = equipos_clasificados_df['equipo_local'].tolist()

    # Simular la Champions League usando el método de Monte Carlo
    num_simulaciones = 100
    victorias_equipo = {equipo: 0 for equipo in equipos_clasificados}

    for _ in range(num_simulaciones):
        for _, partido in partidos_df.iterrows():
            equipo_local = partido['equipo_local']
            equipo_visitante = partido['equipo_visitante']
            resultado = simular_partido()
            if resultado == 'local' and equipo_local in equipos_clasificados:
                victorias_equipo[equipo_local] += 1
            elif resultado == 'visitante' and equipo_visitante in equipos_clasificados:
                victorias_equipo[equipo_visitante] += 1

    # Calcular las probabilidades de ganar la Champions League para cada equipo
    total_victorias = sum(victorias_equipo.values())
    probabilidades = {equipo: victorias / total_victorias for equipo, victorias in victorias_equipo.items()}

    # Imprimir las probabilidades
    for equipo, probabilidad in probabilidades.items():
        print(f"Probabilidad de que {equipo} gane la Champions League: {probabilidad:.2%}")



# Guardar las probabilidades en un archivo CSV
#ruta_archivo_probabilidades = 'MONTECARLO/probabilidades.csv'
#probabilidades_df = pd.DataFrame(probabilidades.items(), columns=['equipo', 'probabilidad'])
#probabilidades_df.to_csv(ruta_archivo_probabilidades, index=False)

