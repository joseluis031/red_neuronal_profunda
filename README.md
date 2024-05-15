# red_neuronal_profunda

El link de este repositorio es el siguiente: [GitHub](https://github.com/joseluis031/red_neuronal_profunda.git)
---------------------------------------
Este trabajo hemos conseguido realizar diferentes predicciones del ganador de la UCL con diferentes modelos además de diferentes análisis como bien podrían ser clustering o series temporales.

La explicación de cada gráfica/análisis esta dentro de su respectivo Notebook

## Contenido

### Notebooks
- [analisis.ipynb](#id1)
  
  En este notebook hemos sacado estadísticas interesante de los csv de datos iniciales y las hemos guardado posteriormente en una Carpeta (CSVS STATS)
  
- [analisis_grafico.ipynb](#id2)
  
  En este notebook hemos realizado un análisis y filtrado de datos y hemos obtenido información interesante en forma de gráficas como bien podría ser: BoxPlot, Correlacción, ScatterPlot...
  
- [analisis_madrid.ipynb](#id3)

  En este notebook hemos realizado una comparación de las estadísticas del Real Madrid Vs Resto de equipos

- [modelo.ipynb](#id4)

  En este notebook realizamos nuestro primer modelo del proyecto del cual obtuvimos una precisión muy baja

- [Clustering.ipynb](#id5)

  En este notebook hemos realizado un análisis utilizando todos los tipos diferentes de Clusters, desde K-Means a Agglomerative CLustering

- [Series.ipynb](#id6)

  En este notebook hemos llevado a cabo una serie temporal en busca de obtener la prediccion de goles de los ultimos 10 años yendo paso a paso.

### Predicciones Ganador UCL
- [Carpeta Met-Gaussiano](#id7)

  En esta carpeta, usando un método Gaussiano hemos obtenido una predicción de la eliminatoria UCL y hemos evaluado esa predicción.

- [Carpeta Random-Forest](#id8)

  En esta carpeta, usando Random-Forest hemos obtenido una predicción de la eliminatoria UCL y hemos evaluado esa predicción.

- [Carpeta Red-Neuronal-Convolucional](#id9)

  En esta carpeta, usando una CNN hemos obtenido una predicción de la eliminatoria UCL y hemos evaluado esa predicción.

- [Carpeta Red-Neuronal-Profunda](#id10)

  En esta carpeta, usando una DNN hemos obtenido una predicción de la eliminatoria UCL y hemos evaluado esa predicción.

- [Carpeta Regresion](#id11)

  En esta carpeta, usando un modelo de Regresión hemos obtenido una predicción de la eliminatoria UCL y hemos evaluado esa predicción.

- [Carpeta XGBoost](#id12)

  En esta carpeta, usando XGBoost obtenido una predicción de la eliminatoria UCL y hemos evaluado esa predicción.

- [Carpeta Arima_Comparar](#id20)
 
  En esta carpeta hemos realizado una comparacion mediante series temporales de algunas estadísticas de los equipos:

  Bayern y Barcelona

- [Carpeta MONTECARLO](#id21)

  En esta carpeta encontramos un script con el que hemos calculado las probabilidades de cada equipo clasificado a octavos
  de ganar la UCL
    

### Otros
- Carpeta Imagenes

  Aquí guardamos todas las imagenes que hemos obtenido en los notebooks

-  sacar_imagenes.py

  Aquí se encuentra el codigo utilizado para scar las imagenes de los notebooks

- [menu.py](#id13)

  Aquí se encuentra el manu diseñado para visualizar el programa

- [main.py](#id14)

  A partir de este archivo podemos ejecutar cualquier tipo de codigo del proyecto

- Carpeta CSVS

  Aquí se encuentran los datos con las temporadas con las que hemos trabajado

  ------------------------------------------------

# Predicciones

## Metodo Gaussiano <a name="id7"></a>
### Resultado

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/ccf316da-4e09-44b1-b080-744df6233e7d)


## Random Forest <a name="id8"></a>
### Resultado

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/69781a9d-bc1b-4baf-b83e-467ad208fc05)


## CNN <a name="id9"></a>
### Resultado

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/6714e637-a6f6-4650-9572-0199b227919f)


## DNN <a name="id10"></a>
### Resultado

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/0cb0f187-b099-431f-9ccb-397c936a8a67)


## Regresion <a name="id11"></a>
### Resultado

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/601b3f26-6084-4e8e-8ae0-605063b53709)


## XGBoost <a name="id12"></a>
### Resultado

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/6faba7af-0831-4205-9fd5-bd71903d129f)



# Notebooks

## Análisis <a name="id1"></a>

```
equipo,victorias_esta_temp,empates_esta_temp,derrotas_esta_temp,porcentaje_victorias_esta_temp,porcentaje_empates_esta_temp,porcentaje_derrotas_esta_temp,victorias_ult10_temp,empates_ult10_temp,derrotas_ult10_temp,porcentaje_victorias_ult10_temp,porcentaje_empates_ult10_temp,porcentaje_derrotas_ult10_temp,Goles Marcados Local_esta_temp,Goles Marcados Visitante_esta_temp,Goles Recibidos Local_esta_temp,Goles Recibidos Visitante_esta_temp,Goles Marcados Local_ult10_temp,Goles Marcados Visitante_ult10_temp,Goles Recibidos Local_ult10_temp,Goles Recibidos Visitante_ult10_temp,Golesxpartido Local_ult10_temp,Golesxpartido Visitante_ult10_temp,Golesxpartido Recibidos Local_ult10_temp,Golesxpartido Recibidos Visitante_ult10_temp,titulos_UCL_ganados,titulos_UCL_ult10_temp
Copenhague,2.0,2.0,2.0,33.3,33.3,33.3,2.0,2.0,2.0,33.3,33.3,33.3,6,2,5,3,6.0,2.0,5.0,3.0,2.0,0.7,1.7,1.0,0,0
Leipzig,4.0,0.0,2.0,66.7,0.0,33.3,22.0,5.0,19.0,47.8,10.9,41.3,6,7,5,5,44.0,37.0,37.0,48.0,1.8,1.7,1.5,2.2,0,0
PSG,2.0,2.0,2.0,33.3,33.3,33.3,30.0,8.0,16.0,55.6,14.8,29.6,6,3,1,7,65.0,56.0,20.0,40.0,2.4,2.1,0.7,1.5,0,0
Lazio,3.0,1.0,2.0,50.0,16.7,33.3,5.0,5.0,4.0,35.7,35.7,28.6,4,3,1,6,13.0,7.0,9.0,11.0,1.9,1.0,1.3,1.6,0,0
PSV,2.0,3.0,1.0,33.3,50.0,16.7,2.0,3.0,1.0,33.3,50.0,16.7,4,4,3,7,4.0,4.0,3.0,7.0,1.3,1.3,1.0,2.3,0,0
Inter,3.0,3.0,0.0,50.0,50.0,0.0,17.0,11.0,11.0,43.6,28.2,28.2,3,5,1,4,24.0,29.0,17.0,24.0,1.3,1.4,0.9,1.2,3,0
Porto,4.0,0.0,2.0,66.7,0.0,33.3,17.0,5.0,15.0,45.9,13.5,40.5,7,8,4,4,30.0,28.0,31.0,22.0,1.6,1.6,1.6,1.2,0,0
Napoles,3.0,1.0,2.0,50.0,16.7,33.3,10.0,2.0,4.0,62.5,12.5,25.0,5,5,4,5,20.0,16.0,8.0,9.0,2.5,2.0,1.0,1.1,0,0
Bayern,5.0,1.0,0.0,83.3,16.7,0.0,47.0,7.0,5.0,79.7,11.9,8.5,6,6,4,2,79.0,82.0,19.0,30.0,2.8,2.6,0.7,1.0,6,2
Real Sociedad,3.0,3.0,0.0,50.0,50.0,0.0,3.0,3.0,0.0,50.0,50.0,0.0,4,3,2,0,4.0,3.0,2.0,0.0,1.3,1.0,0.7,0.0,0,0
Manchester City,6.0,0.0,0.0,100.0,0.0,0.0,43.0,10.0,10.0,68.3,15.9,15.9,9,9,3,4,83.0,62.0,27.0,29.0,2.6,2.0,0.8,0.9,1,1
Barcelona,4.0,0.0,2.0,66.7,0.0,33.3,24.0,9.0,12.0,53.3,20.0,26.7,9,3,2,4,49.0,26.0,34.0,23.0,2.2,1.1,1.5,1.0,5,2
Arsenal,4.0,1.0,1.0,66.7,16.7,16.7,4.0,1.0,1.0,66.7,16.7,16.7,12,4,0,4,12.0,4.0,0.0,4.0,4.0,1.3,0.0,1.3,0,0
Atletico Madrid,4.0,2.0,0.0,66.7,33.3,0.0,14.0,12.0,11.0,37.8,32.4,29.7,11,6,2,4,32.0,19.0,19.0,21.0,1.5,1.2,0.9,1.3,0,0
Dortmund,3.0,2.0,1.0,50.0,33.3,16.7,18.0,10.0,15.0,41.9,23.3,34.9,3,4,1,3,33.0,30.0,18.0,39.0,1.6,1.4,0.9,1.8,1,0
Real Madrid,6.0,0.0,0.0,100.0,0.0,0.0,41.0,9.0,14.0,64.1,14.1,21.9,8,8,2,5,80.0,55.0,38.0,40.0,2.5,1.7,1.2,1.2,14,5
```

## Análisis Gráfico <a name="id2"></a>

### BoxPlot con las victorias y goles de los equipos
![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/e1033f82-072e-4166-878e-7eb7a7130863)
------------------------------------------------
### HeatMap para ver la correlacion entre las estadisticas de los equipos
![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/a27670d6-c872-4ac8-9b3f-bc44faa08d3d)

------------------------------------------------
### ScatterPlot de las victorias de los equipos en los ult10 temp y el % de victorias

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/50a4ac53-92a7-49f0-880c-d2d14340bc76)
------------------------------------------------


## Análisis Madrid <a name="id3"></a>

### Gráfico de Densidad con la Comparacion de las victorias de las ultimas 10 temporadas del real madrid con la media de las victorias de los demas equipos

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/c8032f9c-c364-49c2-a884-4b9333d9501b)
-----------------------------------------------------------------------------------
### Gráfico de barras con la Comparacion de titulos UCL ganados del Real Madrid vs la suma de los titulos de todos los clasificados a octavos
![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/69a6c43a-8f27-4729-a2d8-a956d38be85b)
----------------------------------------------------
### Gráfica de barras con la Comparacion en goles del real madrid vs la media de goles del resto de equipos
![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/d5ce3505-01eb-45da-b103-898712eebd11)
------------------------------------------------------------------------


## Modelo <a name="id4"></a>

### Usando Random Forest

- Matriz de confusion
  
  ![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/d339461f-9031-4248-8198-112e79971094)

- Precisión
  ```
   Precisión del modelo: 0.96

  ```

- Curvas ROC

  ![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/73614cd2-d010-49f8-9d27-b123c3439dc8)


### Usando Regresión

- Precisión
  ```
  Precisión del modelo: 0.63
  ```
- Precisión usando hiperparámetros
  ```
  Precisión promedio: 0.6042343883661249
  Precisión promedio con C=1.0: 0.6209366980325064
  ```

## Clustering <a name="id5"></a>
### K-Means

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/d7b24eed-9c4c-4b19-bbf3-9ee96447a055)
----------------------------------

### Mean-Shift

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/c298e148-1be4-4d4a-9174-e49820e2d676)
-----------------------------------

### Mini-Batch K-Means

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/b0d470ff-0396-4511-927a-37db9ca578ed)
-----------------------------------
### DBSCAN

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/af127bb5-db6e-45e7-91c3-358e57663a25)
-----------------------------------

### OPTICS

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/9af362e5-d2c8-4e34-b72c-7c0d4918bdf4)
-----------------------------------

### GMM

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/9afa7497-5143-4dec-b34b-71d89c74cff5)
-----------------------------------

### Hierarchical

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/9cf9cda3-e9e6-4fdc-9582-e9a2d486d1a5)
-----------------------------------

### Dendrograma con AgglomerativeClustering

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/cc112b42-e08c-4212-af4c-f1caed601a0d)
-----------------------------------

## Series <a name="id6"></a>

### Gráfica para obtener el parametro d de la serie temporal

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/d434015c-2a35-4fd7-97ea-fe0e60a01452)
------------------------------------------

### Prueba de Dickey-Fuller para probar estacionalidad

```
Resultados de la prueba de Dickey-Fuller para columna: Close
Test Statistic                        -4.863350
p-value                                0.000041
No Lags Used                           0.000000
Número de observaciones utilizadas    15.000000
Critical Value (1%)                   -3.964443
Critical Value (5%)                   -3.084908
Critical Value (10%)                  -2.681814
dtype: float64
Conclusion:====>
Rechazar la hipótesis nula
Los datos son estacionarios
```
![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/5f50f1d0-1c80-4c13-a79b-947beb3718c3)
---------------------------------
### Gráfica de Dickey-Fuller para obtener el parametro p

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/38840fb9-c63f-4307-b7ab-8db45100224a)

### Modelo ARIMA


![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/974f7ca9-9f95-4915-a1a0-86afa6c6a990)


![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/965a805f-6119-4ae4-a9f7-ab5df425a5fe)
----------------------------------
### Predicción de goles usando Recursive Forecasting Random Forest

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/c443feef-610a-4889-a15d-4564e65ce8d8)
---------------------------------------

### Predicción de goles usando Recursive Forecasting con Gradient Boosting
![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/912c2ff2-0eb5-4889-a4b3-1ff96bb66a6a)
-----------------------------------

### Predicción de goles uando Multivariate Forecasting con Random Forest

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/01f07d40-7f53-406a-8ab5-de51899cf7ba)
-------------------------------------------------------

### Comparacion mediante series <a name="id20"></a>

![image](https://github.com/joseluis031/red_neuronal_profunda/assets/91721888/27dab1b5-6b70-4591-87f6-882999fb3cb6)

## Montecarlo <a name="id21"></a>

```
Probabilidad de que Copenhague gane la Champions League: 1.33%
Probabilidad de que Leipzig gane la Champions League: 8.25%
Probabilidad de que PSG gane la Champions League: 9.76%
Probabilidad de que Lazio gane la Champions League: 2.66%
Probabilidad de que PSV gane la Champions League: 1.41%
Probabilidad de que Inter gane la Champions League: 6.79%
Probabilidad de que Porto gane la Champions League: 6.65%
Probabilidad de que Napoles gane la Champions League: 3.02%
Probabilidad de que Bayern gane la Champions League: 10.78%
Probabilidad de que Real Sociedad gane la Champions League: 1.30%
Probabilidad de que Manchester City gane la Champions League: 11.26%
Probabilidad de que Barcelona gane la Champions League: 8.05%
Probabilidad de que Arsenal gane la Champions League: 1.78%
Probabilidad de que Atletico Madrid gane la Champions League: 7.21%
Probabilidad de que Dortmund gane la Champions League: 7.97%
Probabilidad de que Real Madrid gane la Champions League: 11.77%
```


# Otros

## Menu <a name="id13"></a>

### Menu principal
```
Menu principal
Que deseas realizar/visualizar?
1. Predicciones del torneo(Regresion, Redes Neuronales, XGBoost, Random Forest, Metodo Gaussiano etc.)
2. Clustering de los equipos que estan en octavos de final
3. Comparativas entre equipos(Gráficas, estadísticas, ARIMA, etc.)
4. Series temporales
5. MonteCarlo
6. Salir
------------------------------------
Escribe el número de la opción que deseas realizar:
```
### Menu Predicciones

```
------------------------------------
Escribe el número de la opción que deseas realizar: 1
------------------------------------
Debes seleccionar el modelo de predicción que deseas utilizar:
1. Regresión
2. Red Neuronal Profunda
3. Red Neuonal Convolucional
4. XGBoost
5. Random Forest
6. Método Gaussiano
7. Volver al menú principal
------------------------------------
Escribe el número de la opción que deseas visualizar:
```

### Menu Clustering

```
------------------------------------
Debes seleccionar el modelo de clustering que deseas visualizar:
1. K-Means
2. Mean-Shift
3. Mini-Batch K-Means
4. DBSCAN
5. OPTICS
6. GMM
7. Hierarchical
8. Dendrograma con AgglomerativeClustering
9. Volver al menú principal
------------------------------------
Escribe el número de la opción que deseas visualizar:
```

### Menu Comparativas equipos

```
------------------------------------
Escribe el número de la opción que deseas realizar: 3
------------------------------------
Debes seleccionar el tipo de comparativa que deseas visualizar:
1. Real Madrid vs Resto de equipos(análisis gráfico)
2. Estadísticas de los equipos en octavos de final
3. Barcelona Vs Bayern (Series temporales)
4. Volver al menú principal
------------------------------------
Escribe el número de la opción que deseas visualizar:
```

### Menu Series Temporales

```
------------------------------------
Vamos a realizar un análisis de series temporales  de los goles de los equipos en octavos de final en los últimos 10 años.
Selecciona el tipo de serie temporal que deseas visualizar (se recomienda por orden):
1. Gráfica para obtener el parametro d de la serie temporal
2. Prueba de Dickey-Fuller para comprobar si la serie es estacionaria
3. Gráfica de Dickey-Fuller para obtener el parametro p
4. Modelo ARIMA Ajustado
5. Predicción de goles usando Recursive Forecasting Random Forest
6. Predicción de goles usando Recursive Forecasting con Gradient Boosting
7. Predicción de goles uando Multivariate Forecasting con Random Forest
8. Volver al menú principal
------------------------------------
Escribe el número de la opción que deseas visualizar:
```

## Main <a name="id14"></a>

```
from menu import menu

if __name__ == "__main__":
    menu()

```

