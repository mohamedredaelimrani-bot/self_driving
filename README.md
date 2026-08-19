# Autonomous Miniature Vehicle using Raspberry Pi and CNN

Projet de fin d’études consacré à la conception et au développement d’un véhicule autonome miniature basé sur **Raspberry Pi 4** et la **vision par ordinateur**.

## Fonctionnement

Camera → Preprocessing → CNN → Steering prediction → Motor control → L298N → DC motors

Le modèle CNN estime une valeur continue de direction à partir des images de la piste. Cette valeur est ensuite convertie en commandes **GAUCHE / TOUT DROIT / DROITE** pour piloter les deux groupes de moteurs.

## Technologies utilisées

- Raspberry Pi 4
- Raspberry Pi Camera
- Python
- TensorFlow / Keras
- OpenCV / NumPy
- GPIO Zero
- L298N
- Flask

## Organisation

- `dataset/` : données d’entraînement
- `models/` : modèles CNN entraînés
- `scripts/` : programmes Python
- `outputs/` : résultats et mesures expérimentales
