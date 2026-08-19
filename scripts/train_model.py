import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =========================
# Chemins du projet
# =========================
PROJECT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_DIR / "dataset" / "driving_log.csv"
DATASET_DIR = PROJECT_DIR / "dataset"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Lecture du CSV
# =========================
df = pd.read_csv(CSV_PATH)

print("Nombre total d'images :", len(df))
print(df.head())

# =========================
# Fonction de prétraitement
# =========================
def preprocess_image(image):
    h, w, _ = image.shape

    # Garder surtout la partie utile de la route
    image = image[int(h * 0.25):h, :]

    # Redimensionner comme dans les projets self-driving
    image = cv2.resize(image, (200, 66))

    # Conversion BGR vers RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalisation entre 0 et 1
    image = image / 255.0

    return image

# =========================
# Charger les images
# =========================
images = []
angles = []

for index, row in df.iterrows():
    img_path = DATASET_DIR / row["image_path"]
    angle = row["steering_angle"]

    image = cv2.imread(str(img_path))

    if image is None:
        continue

    image = preprocess_image(image)

    images.append(image)
    angles.append(angle)

X = np.array(images, dtype=np.float32)
y = np.array(angles, dtype=np.float32)

print("Shape X :", X.shape)
print("Shape y :", y.shape)

# =========================
# Division train / validation / test
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

# =========================
# Création du modèle CNN
# =========================
model = Sequential([
    Conv2D(24, (5, 5), strides=(2, 2), activation="relu", input_shape=(66, 200, 3)),
    Conv2D(36, (5, 5), strides=(2, 2), activation="relu"),
    Conv2D(48, (5, 5), strides=(2, 2), activation="relu"),
    Conv2D(64, (3, 3), activation="relu"),
    Conv2D(64, (3, 3), activation="relu"),

    Flatten(),

    Dense(100, activation="relu"),
    Dropout(0.3),

    Dense(50, activation="relu"),
    Dense(10, activation="relu"),

    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="mse",
    metrics=["mae"]
)

model.summary()

# =========================
# Callbacks
# =========================
checkpoint_path = MODELS_DIR / "best_model.h5"

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True
    )
]

# =========================
# Entraînement
# =========================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)

# =========================
# Évaluation
# =========================
test_loss, test_mae = model.evaluate(X_test, y_test)

print("\nRésultat sur test set :")
print("Test Loss MSE :", test_loss)
print("Test MAE :", test_mae)

# =========================
# Sauvegarde finale
# =========================
final_model_path = MODELS_DIR / "self_driving_model.h5"
model.save(str(final_model_path))

print("\nModèle sauvegardé dans :", final_model_path)

# =========================
# Courbe loss
# =========================
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("Courbe d'entraînement du modèle CNN")
plt.xlabel("Epoch")
plt.ylabel("Loss MSE")
plt.legend()
plt.grid(True)

loss_plot_path = OUTPUTS_DIR / "training_loss.png"
plt.savefig(loss_plot_path)
plt.show()

print("Courbe loss sauvegardée dans :", loss_plot_path)

# =========================
# Comparaison réel / prédit
# =========================
y_pred = model.predict(X_test).flatten()

plt.figure(figsize=(10, 5))
plt.plot(y_test[:100], label="Angles réels")
plt.plot(y_pred[:100], label="Angles prédits")
plt.title("Comparaison entre angles réels et angles prédits")
plt.xlabel("Échantillons")
plt.ylabel("Steering angle")
plt.legend()
plt.grid(True)

pred_plot_path = OUTPUTS_DIR / "real_vs_predicted_angles.png"
plt.savefig(pred_plot_path)
plt.show()

print("Graphe réel/prédit sauvegardé dans :", pred_plot_path)