import cv2; 
import mediapipe as mp; 
import pandas as pd;
import os

INPUT_PATH = "../data/raw_videos"; OUTPUT_PATH = "../data/01_landmarks"

# Crear carpetas si no existen
if not os.path.exists(INPUT_PATH): 
    os.makedirs(INPUT_PATH)
    print(f"  Carpeta '{INPUT_PATH}' creada. Por favor, agrega tus videos (.mp4 o .mov) ahí.")
if not os.path.exists(OUTPUT_PATH): 
    os.makedirs(OUTPUT_PATH)

# Verificar que hay videos
if not os.path.exists(INPUT_PATH) or len(os.listdir(INPUT_PATH)) == 0:
    print(f" Error: No hay videos en '{INPUT_PATH}'")
    print(f"   Por favor, agrega archivos .mp4 o .mov en esa carpeta.")
    exit(1)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

videos_procesados = 0
videos_saltados = 0

for video_file in os.listdir(INPUT_PATH):
    if not video_file.lower().endswith(('.mp4', '.mov')): 
        continue
    
    video_path = os.path.join(INPUT_PATH, video_file)
    video_name = os.path.splitext(video_file)[0]
    csv_path = os.path.join(OUTPUT_PATH, f"{video_name}.csv")

    # Verificar si el video ya fue procesado
    if os.path.exists(csv_path):
        print(f"  ✓ Saltando '{video_file}', ya procesado.")
        videos_saltados += 1
        continue
    
    cap = cv2.VideoCapture(video_path)
    print(f"  → Procesando: '{video_file}'")
    
    landmarks_data = []
    frame_idx = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            frame_data = {'frame': frame_idx}
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_data.update({f'x_{i}': lm.x, f'y_{i}': lm.y, f'z_{i}': lm.z, f'v_{i}': lm.visibility})
            landmarks_data.append(frame_data)
        frame_idx += 1
    
    cap.release()
    if landmarks_data:
        pd.DataFrame(landmarks_data).to_csv(csv_path, index=False)
        print(f"  ✓ Landmarks guardados para '{video_file}' ({len(landmarks_data)} frames).")
        videos_procesados += 1
    else:
        print(f"   Advertencia: No se detectaron landmarks en '{video_file}'.")

pose.close()
print("\n" + "="*50)
print(f"✓ Extracción de landmarks completada.")
print(f"  Videos procesados: {videos_procesados}")
print(f"  Videos saltados (ya procesados): {videos_saltados}")
print("="*50)