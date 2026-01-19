import json
import joblib
import random
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "results.xlsx"
MODELS_PATH = BASE_DIR / "models"
MEMORY_PATH = BASE_DIR / "memory"

def crear_archivos_memoria():
    print('Creando archivos')

def asegurar_directorios():
    MEMORY_PATH.mkdir(exist_ok=True)

def conteo_frecuencias(df, frecuencias):
    columnas_numeros = ["n1", "n2", "n3", "n4", "n5"]
    columna_sb = "superbalota"

    for _, row in df.iterrows():
        for col in columnas_numeros:
            valor = row[col]
            if pd.isna(valor):
                continue

            valor = str(int(valor))
            frecuencias["numeros"][valor] = frecuencias["numeros"].get(valor, 0) + 1

        sb = row[columna_sb]
        if pd.isna(sb):
            continue

        sb = str(int(sb))
        frecuencias["superbalota"][sb] = frecuencias["superbalota"].get(sb, 0) + 1

def construccion_transiciones(df):
    print("✔ Actualizando transiciones...")

    transiciones_path = MEMORY_PATH / "transiciones.json"
    transiciones = {}

    df = df.sort_values("fecha").reset_index(drop=True)

    for i in range(len(df) - 1):
        actual = sorted([
            df.loc[i, "n1"],
            df.loc[i, "n2"],
            df.loc[i, "n3"],
            df.loc[i, "n4"],
            df.loc[i, "n5"]
        ])

        siguiente = sorted([
            df.loc[i + 1, "n1"],
            df.loc[i + 1, "n2"],
            df.loc[i + 1, "n3"],
            df.loc[i + 1, "n4"],
            df.loc[i + 1, "n5"]
        ])

        key = "-".join(map(str, actual))
        next_key = "-".join(map(str, siguiente))

        transiciones.setdefault(key, {})
        transiciones[key][next_key] = transiciones[key].get(next_key, 0) + 1

    with open(transiciones_path, "w", encoding="utf-8") as f:
        json.dump(transiciones, f, indent=2)

def actualizacion_metadata(df):
    print("✔ Actualizando metadata...")

    metadata_path = MEMORY_PATH / "metadata.json"

    metadata = {
        "ultima_actualizacion": df["fecha"].max().strftime("%Y-%m-%d"),
        "total_sorteos": len(df),
        "ventana": 1,
        "modelo_numeros": "DecisionTree",
        "modelo_superbalota": "RandomForest",
        "estado": "entrenable",
        "version": "1.0.0"
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def actualizar_memoria():
    print("\n=== ACTUALIZANDO MEMORIA ===")

    asegurar_directorios()

    if not DATA_PATH.exists():
        raise FileNotFoundError("No existe results.xlsx")

    df = pd.read_excel(DATA_PATH)

    if "fecha" not in df.columns:
        raise ValueError("El Excel debe contener la columna 'fecha'")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])

    print(f"✔ Sorteos procesados: {len(df)}")

    # ---------- FRECUENCIAS ----------
    frecuencias_path = MEMORY_PATH / "frecuencias.json"
    if frecuencias_path.exists():
        with open(frecuencias_path, "r", encoding="utf-8") as f:
            frecuencias = json.load(f)
    else:
        frecuencias = {
            "numeros": {str(i): 0 for i in range(1, 44)},
            "superbalota": {str(i): 0 for i in range(1, 17)}
        }

    print("✔ Actualizando frecuencias...")
    conteo_frecuencias(df, frecuencias)

    with open(frecuencias_path, "w", encoding="utf-8") as f:
        json.dump(frecuencias, f, indent=2)

    construccion_transiciones(df) # TRANSICIONES

    actualizacion_metadata(df) # METADATA

    print("✔ Memoria actualizada correctamente\n")

def evaluar_modelos(modelo_num, modelo_sb, X_test, y_num_test, y_sb_test):
    pred_num = modelo_num.predict(X_test)
    pred_sb = modelo_sb.predict(X_test)

    coincidencias = []

    for real, pred in zip(y_num_test.values, pred_num):
        coincidencias.append(len(set(real) & set(pred)))

    score_numeros = sum(coincidencias) / len(coincidencias)
    score_superbalota = (pred_sb == y_sb_test.values).mean()

    return score_numeros + score_superbalota

def es_mejor(nuevo_score, score_actual_path):
    if not score_actual_path.exists():
        return True

    with open(score_actual_path, "r") as f:
        score_actual = float(f.read())

    return nuevo_score > score_actual

def guardar_mejor_modelo(modelo_num, modelo_sb, score):
    joblib.dump(modelo_num, MODELS_PATH / "numeros.pkl")
    joblib.dump(modelo_sb, MODELS_PATH / "superbalota.pkl")

    with open(MODELS_PATH / "score.txt", "w") as f:
        f.write(str(score))

def registrar_training_history(registro):
    history_path = MODELS_PATH / "training_history.json"

    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    history.append(registro)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def configuracion_penalizada(params, umbral=0.5):
    history_path = MODELS_PATH / "training_history.json"
    if not history_path.exists():
        return False

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    for h in history:
        if h["params"] == params and h["score"] < umbral:
            return True

    return False

import random

def generar_params_evolutivos(iteracion):
    if iteracion < 10:  # exploración
        return {
            "max_depth": random.randint(3, 15),
            "min_samples_split": random.randint(2, 10),
            "n_estimators": random.choice([50, 100, 200, 300])
        }
    else:  # refinamiento
        return {
            "max_depth": max(3, (iteracion % 8) + 3),
            "min_samples_split": max(2, (iteracion % 6) + 2),
            "n_estimators": 100 + ((iteracion % 5) * 50)
        }


def entrenar_modelos(iteraciones=200):
    if not DATA_PATH.exists():
        raise FileNotFoundError("No se encontró data/results.xlsx")

    MODELS_PATH.mkdir(exist_ok=True)

    df = pd.read_excel(DATA_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"]).sort_values("fecha")

    df_feat = df.copy()
    df_feat[["n1", "n2", "n3", "n4", "n5"]] = df_feat[
        ["n1", "n2", "n3", "n4", "n5"]
    ].shift(1)

    df_feat = df_feat.dropna().reset_index(drop=True)

    X = df_feat[["n1", "n2", "n3", "n4", "n5"]]
    y_num = df_feat[["n1", "n2", "n3", "n4", "n5"]]
    y_sb = df_feat["superbalota"]

    mejor_score_path = MODELS_PATH / "score.txt"
    mejor_score = float(mejor_score_path.read_text()) if mejor_score_path.exists() else 0.0

    for i in range(iteraciones):
        params = generar_params_evolutivos(i)

        if configuracion_penalizada(params):
            print(f"⏭ Iteración {i+1} | Configuración penalizada")
            continue

        X_train, X_test, y_num_train, y_num_test, y_sb_train, y_sb_test = train_test_split(
            X, y_num,
            y_sb, test_size=0.2,
            random_state=random.randint(0, 10_000),  # 👈 rompe la repetición,
            shuffle=True
        )

        modelo_numeros = DecisionTreeClassifier(
            random_state=i,
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"]
        )
        modelo_numeros.fit(X_train, y_num_train)

        modelo_sb = RandomForestClassifier(
            random_state=i,
            n_estimators=params["n_estimators"]
        )
        modelo_sb.fit(X_train, y_sb_train)

        score = evaluar_modelos(
            modelo_numeros,
            modelo_sb,
            X_test,
            y_num_test,
            y_sb_test
        )

        registro = {
            "iteracion": i + 1,
            "params": params,
            "score": score,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "resultado": "mejorado" if score > mejor_score else "descartado"
        }

        registrar_training_history(registro)

        print(f"Iteración {i+1} | Score: {score}")

        if score > mejor_score:
            guardar_mejor_modelo(modelo_numeros, modelo_sb, score)
            mejor_score = score
            print("✔ Modelo mejorado y guardado")
        else:
            print("✖ Modelo descartado")


def predecir():
    """
    Usa modelos entrenados para predecir:
    - 5 números (1–43)
    - 1 superbalota (1–16)
    """
    modelo_numeros_path = MODELS_PATH / "numeros.pkl"
    modelo_sb_path = MODELS_PATH / "superbalota.pkl"

    if not modelo_numeros_path.exists() or not modelo_sb_path.exists():
        raise RuntimeError("Modelos no entrenados")

    modelo_numeros = joblib.load(modelo_numeros_path)
    modelo_sb = joblib.load(modelo_sb_path)

    # leer último sorteo
    df = pd.read_excel(DATA_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")
    ultimo = df.sort_values("fecha").iloc[-1]

    X_input = [[
        ultimo["n1"],
        ultimo["n2"],
        ultimo["n3"],
        ultimo["n4"],
        ultimo["n5"]
    ]]

    # predicciones
    numeros_pred = modelo_numeros.predict(X_input)[0]
    superbalota_pred = int(modelo_sb.predict(X_input)[0])

    # asegurar números únicos y rango válido
    numeros_pred = sorted(set(map(int, numeros_pred)))

    while len(numeros_pred) < 5:
        numeros_pred.append(numeros_pred[-1] + 1)

    numeros_pred = [n for n in numeros_pred if 1 <= n <= 43][:5]

    superbalota_pred = max(1, min(16, superbalota_pred))

    return {
        "numeros": numeros_pred,
        "superbalota": superbalota_pred
    }
