import os
import pandas as pd


class DataHandler:
    def __init__(self, carpeta, archivo_eliminacion):
        self.carpeta = carpeta
        self.archivo_eliminacion = archivo_eliminacion
    
    def cargar_datos_entrenamiento(self):
        archivos_csv = [os.path.join(self.carpeta, archivo) for archivo in os.listdir(self.carpeta) if archivo.startswith("temp")]
        dataframes = [pd.read_csv(archivo) for archivo in archivos_csv]
        resultados = pd.concat(dataframes, ignore_index=True)
        resultados = resultados.dropna(subset=['goles_equipo_local', 'goles_equipo_visitante'])
        return resultados
    
    def cargar_datos_eliminacion(self, archivo):
        return pd.read_csv(archivo)
