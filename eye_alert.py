import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Webcam
cap = cv2.VideoCapture(0)

# Pontos dos olhos no MediaPipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Função EAR
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

EAR_THRESHOLD = 0.25   # Limite de olho fechado
CLOSED_TIME_ALERT = 1.5  # segundos

start_close_time = None
alert_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            left_eye = np.array([
                [int(face_landmarks.landmark[i].x * w),
                 int(face_landmarks.landmark[i].y * h)]
                for i in LEFT_EYE
            ])

            right_eye = np.array([
                [int(face_landmarks.landmark[i].x * w),
                 int(face_landmarks.landmark[i].y * h)]
                for i in RIGHT_EYE
            ])

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2

            if ear < EAR_THRESHOLD:
                if start_close_time is None:
                    start_close_time = time.time()
                elif time.time() - start_close_time > CLOSED_TIME_ALERT:
                    alert_active = True
            else:
                start_close_time = None
                alert_active = False

            # Desenhar olhos
            for point in np.concatenate((left_eye, right_eye)):
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

            if alert_active:
                cv2.putText(
                    frame,
                    "ALERTA: OLHO FECHADO!",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

    cv2.imshow("Detector de Olhos", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
