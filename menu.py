

def menu():
    while True:
        def menu_champs():
            print("Menu principal")
            print("Que deseas realizar/visualizar?")
            print("1. Predicciones del torneo(Regresion, Redes Neuronales, XGBoost, Random Forest, Metodo Gaussiano etc.)")
            print("2. Clustering de los equipos que estan en octavos de final")
            print("3. Comparativas entre equipos(Gráficas, estadísticas, ARIMA, etc.)")
            print("4. Series temporales")
            print("5. MonteCarlo")
            print("6. Salir")
            print("------------------------------------")
            eleccion = input("Escribe el número de la opción que deseas realizar: ")
            print("------------------------------------")
            return eleccion
        eleccion = menu_champs()
        
        from time import sleep
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
            
        if eleccion == "1":
            while True:
                print("Debes seleccionar el modelo de predicción que deseas utilizar: ")
                print("1. Regresión")
                print("2. Red Neuronal Profunda")
                print("3. Red Neuonal Convolucional")
                print("4. XGBoost")
                print("5. Random Forest")
                print("6. Método Gaussiano")
                print("7. Volver al menú principal")
                print("------------------------------------")
                eleccion_pred = input("Escribe el número de la opción que deseas visualizar: ")
                print("------------------------------------")
                if eleccion_pred == "1":
                    from Regresion.main import main_reg

                    print("Has elegido la prediccion mediante Regresion:")
                    print("Aqui tienes algunas métricas del modelo:")
                    main_reg()                  
                    
                elif eleccion_pred == "2":
                    from Red_Neuronal_Profunda.main import main_DNN
                    
                    print("Has elegido la prediccion mediante Red Neuronal Profunda:")
                    print("Aqui tienes algunas métricas del modelo:")
                    main_DNN()
                    
                elif eleccion_pred == "3":
                    from Red_Neuronal_Convolucional.main import main_CNN

                    print("Has elegido la prediccion mediante Red Neuronal Convolucional:") 
                    print("Aqui tienes algunas métricas del modelo:")
                    main_CNN()

                elif eleccion_pred == "4":
                    from XGBoost.main import main_XGB
                    
                    print("Has elegido la prediccion mediante XGBoost:")
                    print("Aqui tienes algunas métricas del modelo:")
                    main_XGB()
                    
                elif eleccion_pred == "5":
                    from Random_Forest.main import main_rf
                    
                    print("Has elegido la prediccion mediante Random Forest:")
                    print("Aqui tienes algunas métricas del modelo:")
                    main_rf()
                    
                elif eleccion_pred == "6":
                    from Met_Gaussiano.main import main_GS
                    
                    print("Has elegido la prediccion mediante Método Gaussiano:")
                    print("Aqui tienes algunas métricas del modelo:")
                    main_GS()
                    
                elif eleccion_pred == "7":
                    print("Volviendo al menú principal...")
                    break
                else:
                    print("Opción no válida. Inténtalo de nuevo.")
                    print("------------------------------------")
                    sleep(1.5)
                    continue    
                
                                
            
        elif eleccion == "2":
            

            while True:
                print("Debes seleccionar el modelo de clustering que deseas visualizar: ")
                print("1. K-Means")
                print("2. Mean-Shift")
                print("3. Mini-Batch K-Means")
                print("4. DBSCAN")
                print("5. OPTICS")
                print("6. GMM")
                print("7. Hierarchical")
                print("8. Dendrograma con AgglomerativeClustering")
                print("9. Volver al menú principal")
                print("------------------------------------")
                eleccion_clust = input("Escribe el número de la opción que deseas visualizar: ")
                print("------------------------------------")
                if eleccion_clust == "1":
                    
                    print("Has elegido el clustering mediante K-Means:")
                    print("Cargando...")
                    # Ruta de la imagen
                    ruta_imagen = "Imagenes/clustering_images/K-Means.png"

                    # Cargar la imagen
                    imagen = mpimg.imread(ruta_imagen)

                    # Mostrar la imagen
                    plt.imshow(imagen)
                    plt.axis('off')  
                    plt.show()

                    
                elif eleccion_clust == "2":
                    print("Has elegido el clustering mediante Mean-Shift:")
                    print("Cargando...")
                    # Ruta de la imagen
                    ruta_imagen = "Imagenes/clustering_images/Mean-Shift.png"

                    # Cargar la imagen
                    imagen = mpimg.imread(ruta_imagen)

                    # Mostrar la imagen
                    plt.imshow(imagen)
                    plt.axis('off')  
                    plt.show()
                    
                elif eleccion_clust == "3":
                    print("Has elegido el clustering mediante Mini-Batch K-Means:")
                    print("Cargando...")
                    # Ruta de la imagen
                    ruta_imagen = "Imagenes/clustering_images/Mini-Batch K-Means.png"

                    # Cargar la imagen
                    imagen = mpimg.imread(ruta_imagen)

                    # Mostrar la imagen
                    plt.imshow(imagen)
                    plt.axis('off')  
                    plt.show()
                    
                elif eleccion_clust == "4":
                    print("Has elegido el clustering mediante DBSCAN:")
                    print("Cargando...")
                    # Ruta de la imagen
                    ruta_imagen = "Imagenes/clustering_images/DBSCAN.png"

                    # Cargar la imagen
                    imagen = mpimg.imread(ruta_imagen)

                    # Mostrar la imagen
                    plt.imshow(imagen)
                    plt.axis('off')  
                    plt.show()
                    
                elif eleccion_clust == "5":
                    print("Has elegido el clustering mediante OPTICS:")
                    print("Cargando...")
                    # Ruta de la imagen
                    ruta_imagen = "Imagenes/clustering_images/OPTICS.png"

                    # Cargar la imagen
                    imagen = mpimg.imread(ruta_imagen)

                    # Mostrar la imagen
                    plt.imshow(imagen)
                    plt.axis('off')  
                    plt.show()
                    
                elif eleccion_clust == "6":
                    print("Has elegido el clustering mediante GMM:")
                    print("Cargando...")
                    # Ruta de la imagen
                    ruta_imagen = "Imagenes/clustering_images/GMM.png"

                    # Cargar la imagen
                    imagen = mpimg.imread(ruta_imagen)

                    # Mostrar la imagen
                    plt.imshow(imagen)
                    plt.axis('off')  
                    plt.show()
                    
                elif eleccion_clust == "7":
                    print("Has elegido el clustering mediante Hierarchical:")
                    print("Cargando...")
                    # Ruta de la imagen
                    ruta_imagen = "Imagenes/clustering_images/Hierarchical.png"

                    # Cargar la imagen
                    imagen = mpimg.imread(ruta_imagen)

                    # Mostrar la imagen
                    plt.imshow(imagen)
                    plt.axis('off')  
                    plt.show()
                    
                elif eleccion_clust == "8":
                    print("Has elegido el clustering mediante Dendrograma con AgglomerativeClustering:")
                    print("Cargando...")
                    # Ruta de la imagen
                    ruta_imagen = "Imagenes/clustering_images/dendograma.png"

                    # Cargar la imagen
                    imagen = mpimg.imread(ruta_imagen)

                    # Mostrar la imagen
                    plt.imshow(imagen)
                    plt.axis('off')  
                    plt.show()
                    
                elif eleccion_clust == "9":
                    print("Volviendo al menú principal...")
                    break
                else:
                    print("Opción no válida. Inténtalo de nuevo.")
                    print("------------------------------------")
                    sleep(1.5)
                    continue
            
        elif eleccion == "3":
            while True:
                print("Debes seleccionar el tipo de comparativa que deseas visualizar: ")
                print("1. Real Madrid vs Resto de equipos(análisis gráfico)")
                print("2. Estadísticas de los equipos en octavos de final")
                print("3. Barcelona Vs Bayern (Series temporales)")
                
                print("4. Volver al menú principal")
                print("------------------------------------")
                eleccion_comp = input("Escribe el número de la opción que deseas visualizar: ")
                print("------------------------------------")
                if eleccion_comp == "1":
                    while True:
                        print("Has elegido la comparativa del Real Madrid VS Resto de equipos de octavos :")
                        print("¿Qué deseas visualizar?")
                        print("1. Gráfico de Densidad con la Comparacion de las victorias de las ultimas 10 temporadas"
                            "\n del real madrid con la media de las victorias de los demas equipos")
                        print("2. Gráfico de barras con la Comparacion de titulos UCL ganados "
                            "\nel Real Madrid vs la suma de los titulos de todos los clasificados a octavos")
                        print("3. Gráfica de barras con la Comparacion en goles"
                            "\ndel real madrid vs la media de goles del resto de equipos")
                        
                        print("4. Volver al menú principal")
                        print("------------------------------------")
                        eleccion_comp_RM = input("Escribe el número de la opción que deseas visualizar: ")
                        print("------------------------------------")
                        if eleccion_comp_RM == "1":
                            print("Has elegido la comparativa mediante gráfico de densidad:")
                            print("Cargando...")
                            # Ruta de la imagen
                            ruta_imagen = "Imagenes/analisis_madrid_images/distr_vic_ult10_temp.png"

                            # Cargar la imagen
                            imagen = mpimg.imread(ruta_imagen)

                            # Mostrar la imagen
                            plt.imshow(imagen)
                            plt.axis('off')  
                            plt.show()
                            
                        elif eleccion_comp_RM == "2":
                            print("Has elegido la comparativa de titulos UCL:")
                            print("Cargando...")
                            # Ruta de la imagen
                            ruta_imagen = "Imagenes/analisis_madrid_images/titulos.png"

                            # Cargar la imagen
                            imagen = mpimg.imread(ruta_imagen)

                            # Mostrar la imagen
                            plt.imshow(imagen)
                            plt.axis('off')  
                            plt.show()
                            
                        elif eleccion_comp_RM == "3":
                            print("Has elegido la comparativa en goles:")
                            print("Cargando...")
                            # Ruta de la imagen
                            ruta_imagen = "Imagenes/analisis_madrid_images/goles_ult10_temp.png"

                            # Cargar la imagen
                            imagen = mpimg.imread(ruta_imagen)

                            # Mostrar la imagen
                            plt.imshow(imagen)
                            plt.axis('off')  
                            plt.show()
                            
                        elif eleccion_comp_RM == "4":
                            print("Volviendo al menú principal...")
                            break
                        
                        else:
                            print("Opción no válida. Inténtalo de nuevo.")
                            print("------------------------------------")
                            sleep(1.5)
                            continue
                    
                    
                    
                elif eleccion_comp == "2":
                    while True:
                        print("Has elegido las comparativas mediante estadísticas de los 8 equipos de octavos:")
                        print("¿Qué deseas visualizar?")
                        print("1. BoxPlot con las victorias y goles de los equipos")
                        print("2. HeatMap para ver la correlacion entre las estadisticas de los equipos")
                        print("3. ScatterPlot de las victorias de los equipos en los ult10 temp y el % de victorias"
                              "\n de estos equipos en grupos ")
                        print("4. Volver al menú principal")
                        print("------------------------------------")
                        eleccion_comp_oct = input("Escribe el número de la opción que deseas visualizar: ")
                        print("------------------------------------")
                        if eleccion_comp_oct == "1":
                            print("Has elegido el análisis mediante BoxPlot:")
                            print("Cargando...")
                            # Ruta de la imagen
                            ruta_imagen = "Imagenes/analisis_grafico_images/boxplot.png"

                            # Cargar la imagen
                            imagen = mpimg.imread(ruta_imagen)

                            # Mostrar la imagen
                            plt.imshow(imagen)
                            plt.axis('off')  
                            plt.show()
                            
                        elif eleccion_comp_oct == "2":
                            print("Has elegido el análisis mediante HeatMap de la correlación:")
                            print("Cargando...")
                            # Ruta de la imagen
                            ruta_imagen = "Imagenes/analisis_grafico_images/heatmap.png"

                            # Cargar la imagen
                            imagen = mpimg.imread(ruta_imagen)

                            # Mostrar la imagen
                            plt.imshow(imagen)
                            plt.axis('off')  
                            plt.show()
                            
                        elif eleccion_comp_oct == "3":
                            print("Has elegido el análisis mediante ScatterPlot:")
                            print("Cargando...")
                            # Ruta de la imagen
                            ruta_imagen = "Imagenes/analisis_grafico_images/Scatter.png"

                            # Cargar la imagen
                            imagen = mpimg.imread(ruta_imagen)

                            # Mostrar la imagen
                            plt.imshow(imagen)
                            plt.axis('off')  
                            plt.show()
                        
                        elif eleccion_comp_oct == "4":
                            print("Volviendo al menú principal...")
                            break
                        else:
                            print("Opción no válida. Inténtalo de nuevo.")
                            print("------------------------------------")
                            sleep(1.5)
                            continue
                        
                        
                    
                elif eleccion_comp == "3":
                    while True:
                        print("Has elegido la comparativa Barcelona vs Bayern mediante series temporales:")
                        print("¿Qué deseas visualizar(recomendacion por orden)?")
                        print("1. Prediccion de goles del Barcelona vs Bayern en la ultima temporada vs la realidad"
                            "\n utilizando ARIMA")
                        print("2. Prediccion de goles del Barcelona vs Bayern en la ultima temporada vs la realidad"
                            "\n utilizando ARIMA ajustado")
                        print("3. Prediccion de goles del Barcelona vs Bayern en la ultima temporada vs la realidad"
                            "\n utilizando Arima y Multivariable con Random Forest para REDUCIR ERROR")
                        print("4. Volver al menú principal")
                        print("------------------------------------")
                        eleccion_comp_serie = input("Escribe el número de la opción que deseas visualizar: ")
                        print("------------------------------------")
                        if eleccion_comp_serie == "1":
                            print("Has elegido la primera predicción:")
                            print("Cargando...")
                            # Ruta de la imagen
                            ruta_imagen = "Imagenes/ARIMA_comparar_images/pred_ult_tempBarsa.png"
                            ruta_imagen2 = "Imagenes/ARIMA_comparar_images/pred_ult_tempBayern.png"

                            # Cargar la imagen
                            imagen = mpimg.imread(ruta_imagen)
                            imagen2 = mpimg.imread(ruta_imagen2)

                            # Mostrar la imagen
                            plt.imshow(imagen)
                            plt.axis('off')  
                            plt.show()
                            plt.imshow(imagen2)
                            plt.axis('off')
                            plt.show()

                            
                        elif eleccion_comp_serie == "2":
                            print("Has elegido la comparativa mediante ARIMA ajustado:")
                            print("Cargando...")
                            # Ruta de la imagen
                            ruta_imagen = "Imagenes/ARIMA_comparar_images/ajuste_arima.png"

                            # Cargar la imagen
                            imagen = mpimg.imread(ruta_imagen)

                            # Mostrar la imagen
                            plt.imshow(imagen)
                            plt.axis('off')  
                            plt.show()
                            
                        elif eleccion_comp_serie == "3":
                            print("Has elegido la comparativa mediante ARIMA y Random Forest:")
                            print("Cargando...")
                            # Ruta de la imagen
                            ruta_imagen = "Imagenes/ARIMA_comparar_images/multv_RF.png"

                            # Cargar la imagen
                            imagen = mpimg.imread(ruta_imagen)

                            # Mostrar la imagen
                            plt.imshow(imagen)
                            plt.axis('off')  
                            plt.show()
                        
                        elif eleccion_comp_serie == "4":
                            print("Volviendo al menú principal...")
                            break
                        else:
                            print("Opción no válida. Inténtalo de nuevo.")
                            print("------------------------------------")
                            sleep(1.5)
                            continue
                    
                    
                elif eleccion_comp == "4":
                    print("Volviendo al menú principal...")
                    break
                else:
                    print("Opción no válida. Inténtalo de nuevo.")
                    print("------------------------------------")
                    sleep(1.5)
                    continue
                
        elif eleccion == "4":
            print("Opcion 4")
                        
        elif eleccion == "5":
            print("Opcion 5")
            
            
        elif eleccion == "6":
            print("Saliendo del programa...")
            break
        
        else:
            print("Opción no válida. Inténtalo de nuevo.")
            print("------------------------------------")
            continue
        
menu()