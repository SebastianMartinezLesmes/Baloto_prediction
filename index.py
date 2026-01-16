from bot.bot import crear_archivos_memoria, actualizar_memoria, entrenar_modelos, predecir

def mostrar_menu():
    print("""
=== BALOTO PREDICTION ===
0. Crear archivos de memoria
1. Actualizar memoria
2. Entrenar modelos
3. Predecir próximo sorteo
4. Salir
""")

def main():
    while True:
        mostrar_menu()
        opcion = input("Seleccione una opción: ")

        if opcion == "0":
            crear_archivos_memoria()
        
        if opcion == "1":
            actualizar_memoria()

        elif opcion == "2":
            entrenar_modelos()

        elif opcion == "3":
            resultado = predecir()
            print("Predicción: \n", resultado)

        elif opcion == "4":
            print("Saliendo...")
            break

        else:
            print("Opción inválida")

if __name__ == "__main__":
    main()
