

def menu():
    while True:
        def menu_champs():
            print("Menu principal")
            print("Que deseas realizar/visualizar?")
            print("1. Predicciones del torneo(Regresion, Redes Neuronales, XGBoost, Random Forest, Metodo Gaussiano etc.)")
            print("2. Clustering de los equipos que estan en octavos de final")
            print("3. Comparativas entre equipos(Gráficas, estadísticas, series temporales, etc.)")
            print("4. MonteCarlo")
            print("5. Salir")
            print("------------------------------------")
            eleccion = input("Escribe el número de la opción que deseas realizar: ")
            print("------------------------------------")
            return eleccion
        eleccion = menu_champs()

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
                    continue    
                
                                
            
        elif eleccion == "2":
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

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
                    ruta_imagen = "clustering_images/K-Means.png"

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
                    ruta_imagen = "clustering_images/Mean-Shift.png"

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
                    ruta_imagen = "clustering_images/Mini-Batch K-Means.png"

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
                    ruta_imagen = "clustering_images/DBSCAN.png"

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
                    ruta_imagen = "clustering_images/OPTICS.png"

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
                    ruta_imagen = "clustering_images/GMM.png"

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
                    ruta_imagen = "clustering_images/Hierarchical.png"

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
                    ruta_imagen = "clustering_images/dendograma.png"

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
                    continue
            
        elif eleccion == "3":
            print("Opcion 3")
            
        elif eleccion == "4":
            print("Opcion 4")
            
            
        elif eleccion == "5":
            print("Saliendo del programa...")
            break
        
        else:
            print("Opción no válida. Inténtalo de nuevo.")
            print("------------------------------------")
            continue
        
menu()