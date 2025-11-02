import cv2
import mediapipe as mp

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Inicializar MediaPipe Drawing para dibujar puntos y líneas
mp_drawing = mp.solutions.drawing_utils

# Capturar video de la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB (MediaPipe usa RGB en lugar de BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe Pose
    results = pose.process(rgb_frame)

    # Si se detectan poses, dibujar los puntos clave en la imagen
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    # Mostrar la imagen con los puntos clave
    cv2.imshow("Pose Detection", frame)

    # Presionar 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
