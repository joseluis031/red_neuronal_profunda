

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
                    
                    exit()
                elif eleccion_pred == "2":
                    
                    print("Opcion 2")
                elif eleccion_pred == "3":
                    print("Opcion 3")
                elif eleccion_pred == "4":
                    print("Opcion 4")
                elif eleccion_pred == "5":
                    print("Opcion 5")
                elif eleccion_pred == "6":
                    print("Opcion 6")
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