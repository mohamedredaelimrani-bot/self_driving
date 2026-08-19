from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =========================
# Chemins du projet
# =========================
PROJECT_DIR = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_DIR / "dataset" / "driving_log.csv"
MODEL_PATH = PROJECT_DIR / "models" / "best_model.h5"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Prétraitement identique
# à train_model.py
# =========================
def preprocess_image(image):
    h, w, _ = image.shape

    # Supprimer les 25 % supérieurs
    image = image[int(h * 0.25):h, :]

    # Format attendu par le CNN
    image = cv2.resize(image, (200, 66))

    # OpenCV : BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalisation [0, 1]
    image = image.astype(np.float32) / 255.0

    return image


# =========================
# Charger le CSV
# =========================
df = pd.read_csv(CSV_PATH)

print("Colonnes détectées :", list(df.columns))
print("Nombre total d'échantillons :", len(df))


if "image_path" not in df.columns:
    raise ValueError(
        "La colonne 'image_path' n'existe pas dans driving_log.csv"
    )

if "steering_angle" not in df.columns:
    raise ValueError(
        "La colonne 'steering_angle' n'existe pas dans driving_log.csv"
    )


# =========================
# Reproduire exactement
# la séparation 70 / 15 / 15
# =========================
indices = np.arange(len(df))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42
)

print()
print("Répartition obtenue :")
print("Train      :", len(train_idx))
print("Validation :", len(val_idx))
print("Test       :", len(test_idx))


# =========================
# Charger seulement
# les images de test
# =========================
X_test = []
y_test = []

for idx in test_idx:

    image_path = Path(str(df.loc[idx, "image_path"]))

    # Si le CSV contient un chemin relatif
    if not image_path.is_absolute():
      image_path = PROJECT_DIR / "dataset" / image_path

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Impossible de lire l'image : {image_path}"
        )

    image = preprocess_image(image)

    X_test.append(image)
    y_test.append(float(df.loc[idx, "steering_angle"]))


X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)


print()
print("Forme de X_test :", X_test.shape)
print("Forme de y_test :", y_test.shape)


# =========================
# Charger best_model.h5
# =========================
print()
print("Chargement du modèle :", MODEL_PATH)

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)


# =========================
# Prédictions
# =========================
print("Prédiction sur l'ensemble de test...")

y_pred = model.predict(
    X_test,
    batch_size=32,
    verbose=1
).reshape(-1)


# =========================
# Métriques
# =========================
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)


print()
print("=" * 45)
print("RESULTATS DU MODELE SUR L'ENSEMBLE DE TEST")
print("=" * 45)

print(f"Test Loss MSE : {mse:.6f}")
print(f"Test MAE      : {mae:.6f}")
print(f"Test RMSE     : {rmse:.6f}")

print("=" * 45)


# =========================
# Sauvegarder les résultats
# =========================
results_path = OUTPUT_DIR / "evaluation_best_model.txt"

with open(results_path, "w", encoding="utf-8") as f:
    f.write("Evaluation de best_model.h5\n")
    f.write("===========================\n")
    f.write(f"Nombre total d'images : {len(df)}\n")
    f.write(f"Train : {len(train_idx)}\n")
    f.write(f"Validation : {len(val_idx)}\n")
    f.write(f"Test : {len(test_idx)}\n\n")
    f.write(f"Test Loss MSE : {mse:.6f}\n")
    f.write(f"Test MAE : {mae:.6f}\n")
    f.write(f"Test RMSE : {rmse:.6f}\n")


# =========================
# Figure réel / prédit
# =========================
n = min(100, len(y_test))

plt.figure(figsize=(12, 5))

plt.plot(
    range(n),
    y_test[:n],
    label="Valeur réelle"
)

plt.plot(
    range(n),
    y_pred[:n],
    label="Valeur prédite"
)

plt.xlabel("Échantillon")
plt.ylabel("Valeur de direction")
plt.title(
    "Comparaison entre les valeurs réelles et prédites "
    "sur l'ensemble de test"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

figure_path = (
    OUTPUT_DIR / "real_vs_predicted_angles.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("Résultats sauvegardés dans :")
print(results_path)

print("Figure sauvegardée dans :")
print(figure_path)