import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "results.xlsx"
MODELS_PATH = BASE_DIR / "models"
MEMORY_PATH = BASE_DIR / "memory"


ARCHIVOS_MEMORIA = {
    "frecuencias.json": {
        "numeros": {str(i): 0 for i in range(1, 44)},
        "superbalota": {str(i): 0 for i in range(1, 17)}
    },
    "metadata.json": {
        "ultima_actualizacion": None,
        "total_sorteos": 0,
        "ventana": 1,
        "modelo_numeros": None,
        "modelo_superbalota": None,
        "estado": "inicial",
        "version": "1.0.0"
    },
    "transiciones.json": {}
}

def asegurar_carpeta_memoria():
    MEMORY_PATH.mkdir(exist_ok=True)

def contar_archivos_memoria():
    if not MEMORY_PATH.exists():
        return 0

    return len([
        f for f in MEMORY_PATH.iterdir()
        if f.is_file() and f.name in ARCHIVOS_MEMORIA
    ])

def crear_archivos_faltantes():
    creados = 0

    for nombre, contenido in ARCHIVOS_MEMORIA.items():
        path = MEMORY_PATH / nombre

        if not path.exists():
            with open(path, "w", encoding="utf-8") as f:
                json.dump(contenido, f, indent=2)
            creados += 1

    return creados

def crear_archivos_memoria():
    print("🔍 Verificando memoria...")

    asegurar_carpeta_memoria()

    existentes = contar_archivos_memoria()

    if existentes == len(ARCHIVOS_MEMORIA):
        print(f"✔ Memoria completa ({existentes} archivos)")
        return existentes

    creados = crear_archivos_faltantes()
    total = contar_archivos_memoria()

    print(f"✔ Archivos creados: {creados}")
    print(f"✔ Total en memoria: {total}")

    return total

# ================== TEST ==================
if __name__ == "__main__":
    total=crear_archivos_memoria()
    print(f"📦 Archivos de memoria disponibles: {total}")
