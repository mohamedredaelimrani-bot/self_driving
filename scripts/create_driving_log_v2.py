import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

PROJECT_DIR = Path(__file__).resolve().parents[1]

IMAGES_DIR = PROJECT_DIR / "dataset" / "images"
CSV_PATH = PROJECT_DIR / "dataset" / "driving_log.csv"
DEBUG_DIR = PROJECT_DIR / "dataset" / "debug_labels"
BAD_DIR = PROJECT_DIR / "dataset" / "bad_images"

DEBUG_DIR.mkdir(parents=True, exist_ok=True)
BAD_DIR.mkdir(parents=True, exist_ok=True)

# Paramètres améliorés
BLACK_THRESHOLD = 130       # plus grand = détecte plus de zones sombres
ROI_START_RATIO = 0.20      # garder à partir de 20% de la hauteur
ALLOW_ONE_LINE = True       # accepter aussi les images avec une seule ligne noire
ROAD_WIDTH_RATIO = 0.75     # estimation de largeur de piste si une seule ligne est visible

image_files = sorted([
    f for f in IMAGES_DIR.iterdir()
    if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

print("Nombre total d'images trouvées :", len(image_files))

data = []
bad_count = 0

def compute_steering_angle(image):
    h, w, _ = image.shape

    roi_y_start = int(h * ROI_START_RATIO)
    roi = image[roi_y_start:h, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Flou léger pour réduire le bruit
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détecter les zones noires
    mask = cv2.inRange(gray, 0, BLACK_THRESHOLD)

    # Nettoyage du masque
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    ys, xs = np.where(mask > 0)

    if len(xs) < 200:
        return None, mask, None, None, None, "no_black_pixels"

    center_image = w / 2

    left_pixels = xs[xs < center_image]
    right_pixels = xs[xs >= center_image]

    road_width = w * ROAD_WIDTH_RATIO

    left_x = None
    right_x = None
    method = None

    # Cas idéal : deux lignes visibles
    if len(left_pixels) >= 80 and len(right_pixels) >= 80:
        left_x = np.percentile(left_pixels, 95)
        right_x = np.percentile(right_pixels, 5)
        road_center = (left_x + right_x) / 2
        method = "two_lines"

    # Cas acceptable : seulement ligne gauche visible
    elif ALLOW_ONE_LINE and len(left_pixels) >= 80:
        left_x = np.percentile(left_pixels, 95)
        right_x = left_x + road_width
        road_center = (left_x + right_x) / 2
        method = "left_line_only"

    # Cas acceptable : seulement ligne droite visible
    elif ALLOW_ONE_LINE and len(right_pixels) >= 80:
        right_x = np.percentile(right_pixels, 5)
        left_x = right_x - road_width
        road_center = (left_x + right_x) / 2
        method = "right_line_only"

    else:
        return None, mask, None, None, None, "not_enough_line_pixels"

    offset = (road_center - center_image) / center_image
    steering_angle = np.clip(offset, -1.0, 1.0)

    return steering_angle, mask, left_x, right_x, road_center, method

for image_path in image_files:
    image = cv2.imread(str(image_path))

    if image is None:
        bad_count += 1
        continue

    angle, mask, left_x, right_x, road_center, method = compute_steering_angle(image)

    if angle is None:
        bad_count += 1

        if bad_count <= 200:
            shutil.copy(str(image_path), str(BAD_DIR / image_path.name))

        continue

    relative_path = f"images/{image_path.name}"

    data.append({
        "image_path": relative_path,
        "steering_angle": float(angle),
        "method": method
    })

    # Sauvegarder des images debug
    if len(data) <= 200:
        debug_img = image.copy()
        h, w, _ = debug_img.shape

        # Bleu : centre image
        cv2.line(debug_img, (int(w / 2), 0), (int(w / 2), h), (255, 0, 0), 3)

        # Vert : bords estimés
        cv2.line(debug_img, (int(left_x), 0), (int(left_x), h), (0, 255, 0), 3)
        cv2.line(debug_img, (int(right_x), 0), (int(right_x), h), (0, 255, 0), 3)

        # Rouge : centre de piste estimé
        cv2.line(debug_img, (int(road_center), 0), (int(road_center), h), (0, 0, 255), 3)

        cv2.putText(
            debug_img,
            f"angle={angle:.2f} | {method}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imwrite(str(DEBUG_DIR / image_path.name), debug_img)

df = pd.DataFrame(data)
df.to_csv(CSV_PATH, index=False)

print("\nCréation terminée.")
print("Images acceptées :", len(df))
print("Images rejetées :", bad_count)
print("CSV sauvegardé dans :", CSV_PATH)
print("Images debug dans :", DEBUG_DIR)
print("Images mauvaises dans :", BAD_DIR)

if len(df) > 0:
    print("\nExemple du fichier CSV :")
    print(df.head())

    print("\nRépartition des méthodes :")
    print(df["method"].value_counts())