import os
import pandas as pd


class DataHandler:
    def __init__(self, carpeta_entrenamiento, archivo_prediccion):
        self.carpeta_entrenamiento = carpeta_entrenamiento
        self.archivo_prediccion = archivo_prediccion
    
    def cargar_datos_entrenamiento(self):
        archivos_csv = [os.path.join(self.carpeta_entrenamiento, archivo) for archivo in os.listdir(self.carpeta_entrenamiento) if archivo.startswith("temp")]
        dataframes = []
        for archivo in archivos_csv:
            df = pd.read_csv(archivo)
            dataframes.append(df)
        resultados = pd.concat(dataframes, ignore_index=True)
        resultados = resultados.dropna(subset=['goles_equipo_local', 'goles_equipo_visitante'])
        return resultados
    
    def cargar_datos_prediccion(self):
        return pd.read_csv(self.archivo_prediccion)
