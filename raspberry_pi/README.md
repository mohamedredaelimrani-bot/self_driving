# Raspberry Pi deployment

Ce dossier contient les fichiers utilisés pour le fonctionnement embarqué
du véhicule autonome miniature sur Raspberry Pi 4.

## Fichiers principaux

- `autonomous_drive_web.py` : programme principal de conduite autonome.
  Il assure l'acquisition caméra, le prétraitement, l'inférence du CNN,
  la commande des moteurs, l'interface Flask et l'enregistrement des mesures.

- `autonomous-car.service` : service systemd utilisé pour lancer
  automatiquement l'application sur Raspberry Pi.

Le programme utilise le modèle entraîné `best_model.h5`,
disponible dans le dossier `models/`.
