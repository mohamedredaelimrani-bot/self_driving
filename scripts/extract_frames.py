import cv2
from pathlib import Path

# Dossier principal du projet
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Dossiers
VIDEOS_DIR = PROJECT_DIR / "videos"
IMAGES_DIR = PROJECT_DIR / "dataset" / "images"

# Créer le dossier images s'il n'existe pas
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Sauvegarder une image chaque 3 frames
save_every = 3

# Extensions acceptées
video_extensions = [".mp4", ".mov", ".avi", ".mkv"]

# Chercher les vidéos
video_files = [
    f for f in VIDEOS_DIR.iterdir()
    if f.suffix.lower() in video_extensions
]

if not video_files:
    print("Aucune vidéo trouvée dans le dossier videos.")
    print("Vérifie que tes vidéos sont bien dans :", VIDEOS_DIR)
    exit()

print(f"Nombre de vidéos trouvées : {len(video_files)}")

total_saved = 0

for video_index, video_path in enumerate(video_files, start=1):
    print(f"\nTraitement de la vidéo : {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Erreur : impossible de lire {video_path.name}")
        continue

    frame_id = 0
    saved_id = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % save_every == 0:
            filename = f"video{video_index:02d}_img_{saved_id:05d}.jpg"
            output_path = IMAGES_DIR / filename
            cv2.imwrite(str(output_path), frame)

            saved_id += 1
            total_saved += 1

        frame_id += 1

    cap.release()
    print(f"Images extraites depuis {video_path.name} : {saved_id}")

print("\nExtraction terminée.")
print(f"Nombre total d'images extraites : {total_saved}")
print(f"Images sauvegardées dans : {IMAGES_DIR}")