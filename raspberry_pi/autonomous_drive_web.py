import csv
import sys
import os
import time
import threading

import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from picamera2 import Picamera2
from gpiozero import Motor
from flask import Flask, Response, jsonify


# Sortie non bufferisee, meme quand systemd redirige stdout/stderr
# vers un fichier. Cela permet de suivre run.log en temps reel.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# =========================
# Chemins
# =========================
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "best_model.h5"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TEST_LOG_DIR = OUTPUT_DIR / "test_logs"


# =========================
# Charger le modele
# =========================
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)


# =========================
# Moteurs L298N selon le cablage reel
# =========================
# Convention retenue dans le code :
# - left_motor  = groupe PHYSIQUE gauche
# - right_motor = groupe PHYSIQUE droit
# - forward()   = avance physique
# - backward()  = recul physique
#
# Le cablage materiel n'est pas modifie.
# On corrige uniquement la correspondance logicielle entre les broches,
# les cotes physiques du vehicule et le sens forward/backward.

# Groupe PHYSIQUE gauche
# Canal L298N : IN3=23, IN4=24, ENB=13
left_motor = Motor(forward=24, backward=23, enable=13)

# Groupe PHYSIQUE droit
# Canal L298N : IN1=17, IN2=22, ENA=12
right_motor = Motor(forward=17, backward=22, enable=12)


# =========================
# Parametres
# =========================
base_speed = 0.50
pivot_forward_speed = 0.90
pivot_reverse_speed = 0.20
action_threshold = 0.08
previous_angle = 0.0
ROTATE_IMAGE_180 = True

# Securite : coupe les moteurs si aucune commande recente.
WATCHDOG_TIMEOUT_S = 1.5

# Performances : moyenne calculee sur une fenetre de 5 secondes.
PERF_WINDOW_S = 5.0


# =========================
# Donnees globales pour la page web
# =========================
latest_frame = None
latest_angle = 0.0
latest_action = "STOP"
latest_left_speed = 0.0
latest_right_speed = 0.0
latest_fps = 0.0
latest_inference_ms = 0.0
latest_control_ms = 0.0
last_update_time = time.monotonic()
running = True

# Le programme demarre volontairement en mode ARRET. La camera, le CNN et
# l'interface restent actifs, mais les moteurs attendent un clic sur START.
driving_enabled = False

# Verrous : le premier protege l'etat et les commandes moteurs ; le second
# protege le fichier CSV utilise par les requetes Flask et la boucle autonome.
control_lock = threading.Lock()
log_lock = threading.Lock()
perf_reset_event = threading.Event()

# Une nouvelle session CSV est ouverte a chaque clic sur START.
test_log_file = None
test_log_writer = None
test_log_path = None
test_started_perf = None
last_test_log_name = None


app = Flask(__name__)


def preprocess_image(image_rgb):
    h, _, _ = image_rgb.shape
    image_rgb = image_rgb[int(h * 0.25):h, :]
    image_rgb = cv2.resize(image_rgb, (200, 66))
    image_rgb = image_rgb / 255.0
    return image_rgb


def angle_to_action(angle):
    if angle < -action_threshold:
        return "GAUCHE"
    elif angle > action_threshold:
        return "DROITE"
    else:
        return "TOUT DROIT"


def angle_to_motor_speeds(angle):
    angle = np.clip(angle, -0.60, 0.60)

    # Convention :
    # vitesse positive = avance physique
    # vitesse negative = recul physique

    # Tout droit : les deux groupes avancent a la meme vitesse.
    if abs(angle) <= action_threshold:
        left_speed = base_speed
        right_speed = base_speed

    # Virage a gauche :
    # groupe gauche en recul faible, groupe droit en avance forte.
    elif angle < 0:
        left_speed = -pivot_reverse_speed
        right_speed = pivot_forward_speed

    # Virage a droite :
    # groupe gauche en avance forte, groupe droit en recul faible.
    else:
        left_speed = pivot_forward_speed
        right_speed = -pivot_reverse_speed

    return float(left_speed), float(right_speed)


def drive(left_speed, right_speed):
    # Convention :
    # vitesse positive = avance physique
    # vitesse negative = recul physique
    # forward() = avance physique
    # backward() = recul physique

    if left_speed >= 0:
        left_motor.forward(left_speed)
    else:
        left_motor.backward(-left_speed)

    if right_speed >= 0:
        right_motor.forward(right_speed)
    else:
        right_motor.backward(-right_speed)


def stop_motors():
    left_motor.stop()
    right_motor.stop()


def _write_test_row_locked(
    event,
    angle,
    action,
    left_speed,
    right_speed,
    fps,
    inference_ms,
    latency_ms,
):
    """Ecrit une ligne CSV. log_lock doit deja etre acquis."""
    if test_log_writer is None or test_log_file is None:
        return

    elapsed_s = 0.0
    if test_started_perf is not None:
        elapsed_s = time.perf_counter() - test_started_perf

    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    test_log_writer.writerow(
        [
            timestamp,
            f"{elapsed_s:.3f}",
            event,
            f"{angle:.6f}",
            action,
            f"{left_speed:.3f}",
            f"{right_speed:.3f}",
            f"{fps:.3f}",
            f"{inference_ms:.3f}",
            f"{latency_ms:.3f}",
        ]
    )
    test_log_file.flush()


def start_test_log():
    """Cree automatiquement un nouveau fichier CSV pour l'essai."""
    global test_log_file, test_log_writer, test_log_path
    global test_started_perf, last_test_log_name

    with log_lock:
        if test_log_file is not None:
            return test_log_path

        new_log_file = None
        try:
            TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)
            file_stamp = datetime.now().astimezone().strftime(
                "%Y%m%d_%H%M%S_%f"
            )
            new_log_path = TEST_LOG_DIR / f"test_{file_stamp}.csv"
            new_log_file = new_log_path.open(
                "x",
                newline="",
                encoding="utf-8",
            )
            new_log_writer = csv.writer(new_log_file, delimiter=";")
            new_log_writer.writerow(
                [
                    "timestamp",
                    "elapsed_s",
                    "event",
                    "smooth_angle",
                    "action",
                    "left_speed",
                    "right_speed",
                    "fps",
                    "inference_ms",
                    "latency_ms",
                ]
            )

            test_log_file = new_log_file
            test_log_writer = new_log_writer
            test_log_path = new_log_path
            test_started_perf = time.perf_counter()
            last_test_log_name = new_log_path.name

            _write_test_row_locked(
                "START",
                latest_angle,
                "STOP",
                0.0,
                0.0,
                latest_fps,
                latest_inference_ms,
                latest_control_ms,
            )
            print(f"[TEST] Nouveau log CSV : {new_log_path}")
            return new_log_path

        except OSError as error:
            if new_log_file is not None:
                new_log_file.close()
            test_log_file = None
            test_log_writer = None
            test_log_path = None
            test_started_perf = None
            print(f"[ERREUR] Impossible de creer le log CSV : {error}")
            return None


def write_test_measure(
    angle,
    action,
    left_speed,
    right_speed,
    fps,
    inference_ms,
    latency_ms,
):
    """Ajoute une mesure dans le CSV actif, toutes les cinq secondes."""
    with log_lock:
        try:
            _write_test_row_locked(
                "MEASURE",
                angle,
                action,
                left_speed,
                right_speed,
                fps,
                inference_ms,
                latency_ms,
            )
        except OSError as error:
            print(f"[ERREUR] Ecriture du log CSV impossible : {error}")


def stop_test_log(event="STOP"):
    """Ajoute l'evenement final, vide le tampon et ferme le CSV actif."""
    global test_log_file, test_log_writer, test_log_path
    global test_started_perf, last_test_log_name

    with log_lock:
        if test_log_file is None:
            return None

        closed_path = test_log_path
        try:
            _write_test_row_locked(
                event,
                latest_angle,
                "STOP",
                0.0,
                0.0,
                latest_fps,
                latest_inference_ms,
                latest_control_ms,
            )
        except OSError as error:
            print(f"[ERREUR] Finalisation du log CSV impossible : {error}")
        finally:
            try:
                test_log_file.close()
            except OSError as error:
                print(f"[ERREUR] Fermeture du log CSV impossible : {error}")

            if closed_path is not None:
                last_test_log_name = closed_path.name
            test_log_file = None
            test_log_writer = None
            test_log_path = None
            test_started_perf = None

        if closed_path is not None:
            print(f"[TEST] Log CSV ferme : {closed_path}")
        return closed_path


def watchdog_loop():
    """Arrete la conduite si la boucle principale ne se met plus a jour."""
    global driving_enabled, latest_action
    global latest_left_speed, latest_right_speed

    while running:
        time.sleep(0.5)
        watchdog_stopped_car = False

        with control_lock:
            if (
                driving_enabled
                and time.monotonic() - last_update_time > WATCHDOG_TIMEOUT_S
            ):
                driving_enabled = False
                stop_motors()
                latest_action = "STOP"
                latest_left_speed = 0.0
                latest_right_speed = 0.0
                stop_test_log("WATCHDOG")
                watchdog_stopped_car = True

        if watchdog_stopped_car:
            print(
                "[WATCHDOG] Pas de mise a jour recente : "
                "conduite desactivee et moteurs arretes."
            )


def autonomous_loop():
    global latest_frame, latest_angle, latest_action
    global latest_left_speed, latest_right_speed
    global latest_fps, latest_inference_ms, latest_control_ms
    global previous_angle, last_update_time, running, driving_enabled

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    # Warm-up : le premier appel a predict() est generalement plus lent.
    # Il est effectue avant les mesures pour ne pas fausser les resultats.
    dummy = np.zeros((1, 66, 200, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    last_update_time = time.monotonic()

    # Compteurs utilises pour calculer les moyennes de performance.
    perf_start = time.perf_counter()
    perf_frames = 0
    perf_inference_sum = 0.0
    perf_control_sum = 0.0

    print("Systeme autonome lance.")
    print("Ouvre le navigateur PC sur : http://IP_RASPBERRY:5000")
    print("Mode initial : ARRET. Clique sur START pour activer les moteurs.")
    print("Ctrl + C pour arreter.")

    try:
        while running:
            # Un clic sur START aligne la premiere moyenne de performance
            # sur le debut reel de la nouvelle session d'essai.
            if perf_reset_event.is_set():
                perf_start = time.perf_counter()
                perf_frames = 0
                perf_inference_sum = 0.0
                perf_control_sum = 0.0
                perf_reset_event.clear()

            # Debut du temps capture -> commande moteur.
            cycle_start = time.perf_counter()

            frame_rgb = picam2.capture_array()

            if ROTATE_IMAGE_180:
                frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_180)

            img = preprocess_image(frame_rgb)
            img = np.expand_dims(img, axis=0)

            # Mesure du temps d'inference du CNN seulement.
            inference_start = time.perf_counter()
            angle = float(model.predict(img, verbose=0)[0][0])
            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000.0

            # 0,4 pour la valeur lissee precedente et 0,6 pour
            # la nouvelle prediction brute du CNN.
            smooth_angle = 0.4 * previous_angle + 0.6 * angle
            previous_angle = smooth_angle

            predicted_action = angle_to_action(smooth_angle)
            predicted_left, predicted_right = angle_to_motor_speeds(
                smooth_angle
            )

            # Le meme verrou est utilise par les routes START/STOP. Ainsi,
            # aucune nouvelle commande moteur ne peut etre envoyee apres que
            # la requete STOP a rendu la main au navigateur.
            with control_lock:
                if driving_enabled:
                    drive(predicted_left, predicted_right)
                    action = predicted_action
                    left_speed = predicted_left
                    right_speed = predicted_right
                else:
                    action = "STOP"
                    left_speed = 0.0
                    right_speed = 0.0

                drive_is_enabled = driving_enabled
                last_update_time = time.monotonic()

            # Temps ecoule entre le debut de la capture et l'envoi
            # de la commande aux moteurs.
            control_ms = (
                time.perf_counter() - cycle_start
            ) * 1000.0

            display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            cv2.putText(
                display_frame,
                f"Angle: {smooth_angle:.2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                display_frame,
                f"Action: {action}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                f"L: {left_speed:.2f}  R: {right_speed:.2f}",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                display_frame,
                (
                    f"FPS: {latest_fps:.2f} | "
                    f"Inference: {latest_inference_ms:.1f} ms | "
                    f"Latence: {latest_control_ms:.1f} ms"
                ),
                (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display_frame,
                f"Mode: {'MARCHE' if drive_is_enabled else 'ARRET'}",
                (30, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if drive_is_enabled else (0, 0, 255),
                2,
            )

            latest_frame = display_frame
            latest_angle = smooth_angle
            latest_action = action
            latest_left_speed = left_speed
            latest_right_speed = right_speed

            # Accumulation des mesures sur une fenetre de 5 secondes.
            perf_frames += 1
            perf_inference_sum += inference_ms
            perf_control_sum += control_ms

            perf_elapsed = time.perf_counter() - perf_start

            if perf_elapsed >= PERF_WINDOW_S:
                fps = perf_frames / perf_elapsed
                average_inference = perf_inference_sum / perf_frames
                average_control = perf_control_sum / perf_frames

                # Valeurs affichees dans le flux video de l'interface web.
                latest_fps = fps
                latest_inference_ms = average_inference
                latest_control_ms = average_control

                # Avec systemd, cette ligne est enregistree dans run.log.
                print(
                    f"[PERF] FPS: {fps:.2f} | "
                    f"Inference: {average_inference:.2f} ms | "
                    f"Capture->commande: {average_control:.2f} ms | "
                    f"Angle: {smooth_angle:.2f} | Action: {action}"
                )

                # Une ligne est ajoutee au CSV seulement lorsqu'un essai
                # a ete demarre depuis l'interface.
                write_test_measure(
                    smooth_angle,
                    action,
                    left_speed,
                    right_speed,
                    fps,
                    average_inference,
                    average_control,
                )

                perf_start = time.perf_counter()
                perf_frames = 0
                perf_inference_sum = 0.0
                perf_control_sum = 0.0

    except Exception as error:
        print(f"[ERREUR] autonomous_loop a plante : {error}")
    finally:
        with control_lock:
            driving_enabled = False
            stop_motors()
            latest_action = "STOP"
            latest_left_speed = 0.0
            latest_right_speed = 0.0
            stop_test_log("SYSTEM_STOP")

        picam2.stop()
        running = False
        print("Systeme arrete.")

        # Le code d'echec demande a systemd de relancer proprement
        # le service si la boucle autonome s'est arretee.
        os._exit(1)


def generate_frames():
    while True:
        if latest_frame is None:
            time.sleep(0.1)
            continue

        ret, buffer = cv2.imencode(".jpg", latest_frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


def get_status_payload():
    """Construit l'etat JSON affiche par l'interface web."""
    with control_lock:
        enabled = driving_enabled
        service_running = running

    with log_lock:
        active_log_name = (
            test_log_path.name if test_log_path is not None else None
        )
        last_log_name = last_test_log_name

    return {
        "service_running": service_running,
        "driving_enabled": enabled,
        "mode": "MARCHE" if enabled else "ARRET",
        "angle": round(latest_angle, 4),
        "action": latest_action,
        "left_speed": round(latest_left_speed, 3),
        "right_speed": round(latest_right_speed, 3),
        "fps": round(latest_fps, 3),
        "inference_ms": round(latest_inference_ms, 3),
        "latency_ms": round(latest_control_ms, 3),
        "log_active": active_log_name is not None,
        "active_log": active_log_name,
        "last_log": last_log_name,
    }


@app.route("/api/status")
def api_status():
    return jsonify(get_status_payload())


@app.route("/api/start", methods=["POST"])
def api_start():
    """Active la conduite et ouvre automatiquement un nouveau log CSV."""
    global driving_enabled, last_update_time, latest_action
    global latest_left_speed, latest_right_speed
    global latest_fps, latest_inference_ms, latest_control_ms

    message = "La conduite est deja active."
    log_created = True

    with control_lock:
        if not running:
            return jsonify(
                {
                    "ok": False,
                    "message": "La boucle autonome n'est pas disponible.",
                }
            ), 503

        if not driving_enabled:
            latest_action = "DEMARRAGE"
            latest_left_speed = 0.0
            latest_right_speed = 0.0
            latest_fps = 0.0
            latest_inference_ms = 0.0
            latest_control_ms = 0.0
            new_log_path = start_test_log()
            log_created = new_log_path is not None
            driving_enabled = True
            last_update_time = time.monotonic()
            perf_reset_event.set()
            message = "Conduite activee."
            print("[CONTROLE] START : conduite activee.")

    payload = get_status_payload()
    payload["ok"] = True
    payload["message"] = message
    if not log_created:
        payload["message"] += " Attention : creation du CSV impossible."
    return jsonify(payload)


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Coupe immediatement les moteurs sans arreter Flask ni systemd."""
    global driving_enabled, latest_action
    global latest_left_speed, latest_right_speed

    with control_lock:
        was_enabled = driving_enabled
        driving_enabled = False
        stop_motors()
        latest_action = "STOP"
        latest_left_speed = 0.0
        latest_right_speed = 0.0
        stop_test_log("STOP")

    if was_enabled:
        message = "Conduite arretee et log CSV ferme."
        print("[CONTROLE] STOP : moteurs arretes.")
    else:
        message = "La conduite est deja arretee."

    payload = get_status_payload()
    payload["ok"] = True
    payload["message"] = message
    return jsonify(payload)


@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Autonomous Car Monitor</title>
        <style>
            body {
                background: #111;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
            }
            h1 { color: #00ff88; }
            .controls {
                margin: 18px auto;
                display: flex;
                justify-content: center;
                gap: 14px;
                flex-wrap: wrap;
            }
            button {
                min-width: 150px;
                padding: 13px 22px;
                border: 0;
                border-radius: 8px;
                color: white;
                font-size: 18px;
                font-weight: bold;
                cursor: pointer;
            }
            button:disabled { opacity: 0.45; cursor: not-allowed; }
            #start-button { background: #009e55; }
            #stop-button { background: #d32f2f; }
            .status-panel {
                width: 90%;
                max-width: 900px;
                margin: 0 auto 18px;
                padding: 12px;
                background: #202020;
                border-radius: 8px;
                font-size: 18px;
            }
            #mode { font-weight: bold; }
            #message { min-height: 24px; color: #ffd54f; }
            img {
                width: 90%;
                max-width: 900px;
                border: 3px solid #00ff88;
                border-radius: 10px;
            }
            .info { margin-top: 15px; font-size: 20px; }
        </style>
    </head>
    <body>
        <h1>Autonomous Car - Live Camera</h1>
        <div class="controls">
            <button id="start-button" onclick="sendCommand('start')">
                START
            </button>
            <button id="stop-button" onclick="sendCommand('stop')">
                STOP
            </button>
        </div>
        <div class="status-panel">
            <div>Etat : <span id="mode">Chargement...</span></div>
            <div>Log CSV : <span id="log-name">Aucun</span></div>
            <div id="message"></div>
        </div>
        <img src="/video_feed">
        <div class="info">Camera + CNN + Commandes moteurs + Mesures</div>

        <script>
            const startButton = document.getElementById("start-button");
            const stopButton = document.getElementById("stop-button");
            const modeElement = document.getElementById("mode");
            const logElement = document.getElementById("log-name");
            const messageElement = document.getElementById("message");

            function updateStatus(data) {
                const enabled = Boolean(data.driving_enabled);
                modeElement.textContent = enabled ? "MARCHE" : "ARRET";
                modeElement.style.color = enabled ? "#00ff88" : "#ff5252";
                startButton.disabled = enabled;
                stopButton.disabled = !enabled;

                const logName = data.active_log || data.last_log || "Aucun";
                logElement.textContent = logName;
            }

            async function refreshStatus() {
                try {
                    const response = await fetch("/api/status", {
                        cache: "no-store"
                    });
                    if (!response.ok) throw new Error("Etat indisponible");
                    updateStatus(await response.json());
                } catch (error) {
                    messageElement.textContent = "Connexion au serveur perdue.";
                }
            }

            async function sendCommand(command) {
                startButton.disabled = true;
                stopButton.disabled = true;
                messageElement.textContent = "Commande en cours...";

                try {
                    const response = await fetch(`/api/${command}`, {
                        method: "POST"
                    });
                    const data = await response.json();
                    updateStatus(data);
                    messageElement.textContent = data.message || "";
                } catch (error) {
                    messageElement.textContent = "Echec de la commande.";
                    await refreshStatus();
                }
            }

            refreshStatus();
            setInterval(refreshStatus, 1000);
        </script>
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    car_thread = threading.Thread(target=autonomous_loop, daemon=True)
    car_thread.start()

    watchdog_thread = threading.Thread(target=watchdog_loop, daemon=True)
    watchdog_thread.start()

    try:
        # threaded=True evite que le flux video bloque les autres requetes.
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True,
        )
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        with control_lock:
            driving_enabled = False
            stop_motors()
            stop_test_log("SYSTEM_STOP")
