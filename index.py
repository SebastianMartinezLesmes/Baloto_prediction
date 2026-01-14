from bot.bot import actualizar_memoria, entrenar_modelos, predecir

def mostrar_menu():
    print("""
=== BALOTO PREDICTION ===
1. Actualizar memoria desde Excel
2. Entrenar modelos
3. Predecir próximo sorteo
4. Salir
""")

def main():
    while True:
        mostrar_menu()
        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            actualizar_memoria()

        elif opcion == "2":
            # aquí luego prepararás X e y reales
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
