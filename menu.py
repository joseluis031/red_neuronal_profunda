

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
            print("Opcion 2")
            
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