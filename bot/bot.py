import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "results.xlsx"
MODELS_PATH = BASE_DIR / "models"
MEMORY_PATH = BASE_DIR / "memory"

def conteo_frecuencias(df):
    frecuencias_path = MEMORY_PATH / "frecuencias.json"
    if frecuencias_path.exists():
        with open(frecuencias_path, "r", encoding="utf-8") as f:
            frecuencias = json.load(f)
    else:
        frecuencias = {
            "numeros": {str(i): 0 for i in range(1, 44)},
            "superbalota": {str(i): 0 for i in range(1, 17)}
        }

    for _, row in df.iterrows():
        for col in ["n1", "n2", "n3", "n4", "n5"]:
            frecuencias["numeros"][str(int(row[col]))] += 1
        frecuencias["superbalota"][str(int(row["superbalota"]))] += 1
    with open(frecuencias_path, "w", encoding="utf-8") as f:
        json.dump(frecuencias, f, indent=2)

def construccion_transiciones(df):
    transiciones_path = MEMORY_PATH / "transiciones.json"
    transiciones = {}
    df = df.sort_values("fecha")
    for i in range(len(df) - 1):
        actual = sorted([
            df.iloc[i]["n1"],
            df.iloc[i]["n2"],
            df.iloc[i]["n3"],
            df.iloc[i]["n4"],
            df.iloc[i]["n5"]
        ])
        siguiente = sorted([
            df.iloc[i + 1]["n1"],
            df.iloc[i + 1]["n2"],
            df.iloc[i + 1]["n3"],
            df.iloc[i + 1]["n4"],
            df.iloc[i + 1]["n5"]
        ])
        key = "-".join(map(str, actual))
        next_key = "-".join(map(str, siguiente))
        if key not in transiciones:
            transiciones[key] = {}

        transiciones[key][next_key] = transiciones[key].get(next_key, 0) + 1
    with open(transiciones_path, "w", encoding="utf-8") as f:
        json.dump(transiciones, f, indent=2)

def actualizacion_metadata(df):
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
    """
    Lee results.xlsx y actualiza:
    - frecuencias.json
    - transiciones.json
    - metadata.json
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError("No se encontró data/results.xlsx")
    df = pd.read_excel(DATA_PATH)
    if "fecha" not in df.columns:
        raise ValueError("El Excel debe contener la columna 'fecha'")

    # Validación estricta del formato de fecha
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")

    conteo_frecuencias(df)
    construccion_transiciones(df)
    actualizacion_metadata(df)
    pass

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

def entrenar_modelos(iteraciones=200):
    if not DATA_PATH.exists():
        raise FileNotFoundError("No se encontró data/results.xlsx")

    df = pd.read_excel(DATA_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")

    X = df[["n1", "n2", "n3", "n4", "n5"]].shift(1).dropna()
    y_num = df[["n1", "n2", "n3", "n4", "n5"]].iloc[1:]
    y_sb = df["superbalota"].iloc[1:]

    split = int(len(X) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_num_train, y_num_test = y_num[:split], y_num[split:]
    y_sb_train, y_sb_test = y_sb[:split], y_sb[split:]

    for i in range(iteraciones):
        modelo_numeros = DecisionTreeClassifier(
            max_depth=None,
            random_state=None
        )

        modelo_sb = RandomForestClassifier(
            n_estimators=100,
            random_state=None
        )

        modelo_numeros.fit(X_train, y_num_train)
        modelo_sb.fit(X_train, y_sb_train)

        score = evaluar_modelos(
            modelo_numeros,
            modelo_sb,
            X_test,
            y_num_test,
            y_sb_test
        )

        if es_mejor(score, MODELS_PATH / "score.txt"):
            guardar_mejor_modelo(modelo_numeros, modelo_sb, score)


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
