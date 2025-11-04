#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface Real-time - Modelo Re-entrenado (Solo Grupo Principal)
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
from pathlib import Path
from collections import deque
import json

# ==================== CONFIGURACIÓN ====================
MODELS_PATH = Path(__file__).parent / "models_retrained"
WINDOW_SIZE = 15  # ~0.5 seg @ 30 FPS
CONFIDENCE_THRESHOLD = 0.4

print("="*60)
print("🎥 INTERFAZ REAL-TIME - MODELO RE-ENTRENADO")
print("="*60)

# ==================== CARGAR MODELOS ====================
print("\n📦 Cargando modelos...")
model = joblib.load(MODELS_PATH / "xgboost_model.pkl")
scaler = joblib.load(MODELS_PATH / "scaler.pkl")
label_encoder = joblib.load(MODELS_PATH / "label_encoder.pkl")

with open(MODELS_PATH / "model_metadata.json", "r") as f:
    metadata = json.load(f)

print(f"✅ Modelo cargado: {len(metadata['class_names'])} clases")
print(f"   Clases: {metadata['class_names']}")
print(f"   Accuracy: {metadata['xgb_accuracy']:.4f}")
print(f"   F1 (macro): {metadata['xgb_f1_macro']:.4f}")

# ==================== MEDIAPIPE ====================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ==================== FUNCIONES ====================

def calculate_angle(a, b, c):
    """Calcula ángulo entre 3 puntos"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def calculate_distance(p1, p2):
    """Distancia euclidiana"""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def extract_features_from_window(window):
    """Extrae features de una ventana de landmarks"""
    if len(window) < WINDOW_SIZE * 0.7:
        return None
    
    left_knee_angles = []
    right_knee_angles = []
    left_hip_angles = []
    right_hip_angles = []
    trunk_incl = []
    hip_dist = []
    hip_heights = []
    knee_heights = []
    ankle_heights = []
    
    for lm in window:
        coords = {
            'left_hip': (lm[23].x, lm[23].y, lm[23].z),
            'right_hip': (lm[24].x, lm[24].y, lm[24].z),
            'left_knee': (lm[25].x, lm[25].y, lm[25].z),
            'right_knee': (lm[26].x, lm[26].y, lm[26].z),
            'left_ankle': (lm[27].x, lm[27].y, lm[27].z),
            'right_ankle': (lm[28].x, lm[28].y, lm[28].z),
            'left_shoulder': (lm[11].x, lm[11].y, lm[11].z),
            'right_shoulder': (lm[12].x, lm[12].y, lm[12].z)
        }
        
        # Centros
        hip_center = (
            (coords['left_hip'][0] + coords['right_hip'][0]) / 2,
            (coords['left_hip'][1] + coords['right_hip'][1]) / 2
        )
        shoulder_center = (
            (coords['left_shoulder'][0] + coords['right_shoulder'][0]) / 2,
            (coords['left_shoulder'][1] + coords['right_shoulder'][1]) / 2
        )
        knee_center_y = (coords['left_knee'][1] + coords['right_knee'][1]) / 2
        ankle_center_y = (coords['left_ankle'][1] + coords['right_ankle'][1]) / 2
        
        # Ángulos de rodillas
        left_knee_angles.append(
            calculate_angle(
                coords['left_hip'][:2],
                coords['left_knee'][:2],
                coords['left_ankle'][:2]
            )
        )
        right_knee_angles.append(
            calculate_angle(
                coords['right_hip'][:2],
                coords['right_knee'][:2],
                coords['right_ankle'][:2]
            )
        )
        
        # Ángulos de cadera (crítico para sentarse)
        left_hip_angles.append(
            calculate_angle(
                shoulder_center[:2],
                coords['left_hip'][:2],
                coords['left_knee'][:2]
            )
        )
        right_hip_angles.append(
            calculate_angle(
                shoulder_center[:2],
                coords['right_hip'][:2],
                coords['right_knee'][:2]
            )
        )
        
        # Inclinación del tronco
        trunk_incl.append(
            np.degrees(np.arctan2(
                shoulder_center[0] - hip_center[0],
                hip_center[1] - shoulder_center[1]
            ))
        )
        
        # Distancia caderas-hombros
        hip_dist.append(calculate_distance(hip_center, shoulder_center))
        
        # Alturas (crítico para sentarse)
        hip_heights.append(hip_center[1])
        knee_heights.append(knee_center_y)
        hip_dist.append(calculate_distance(hip_center, shoulder_center))
        
        # Alturas (crítico para sentarse)
        hip_heights.append(hip_center[1])
        knee_heights.append(knee_center_y)
        ankle_heights.append(ankle_center_y)
    
    features = np.array([
        np.mean(left_knee_angles),
        np.std(left_knee_angles),
        np.mean(right_knee_angles),
        np.std(right_knee_angles),
        np.mean(left_hip_angles),
        np.std(left_hip_angles),
        np.mean(right_hip_angles),
        np.std(right_hip_angles),
        np.mean(trunk_incl),
        np.std(trunk_incl),
        np.mean(hip_dist),
        np.std(hip_dist),
        np.mean(hip_heights),
        np.std(hip_heights),
        np.mean(knee_heights),
        np.std(knee_heights),
        np.mean(ankle_heights),
        np.std(ankle_heights),
        np.mean([abs(h - k) for h, k in zip(hip_heights, knee_heights)]),
        np.mean([abs(k - a) for k, a in zip(knee_heights, ankle_heights)])
    ])
    
    return features


# ==================== MAIN ====================

def main():
    cap = cv2.VideoCapture(0)
    landmarks_buffer = deque(maxlen=WINDOW_SIZE)
    
    print("\n" + "="*60)
    print("🎥 Cámara iniciada")
    print("="*60)
    print("\nControles:")
    print("   - Presiona 'q' para salir")
    print("   - Presiona 's' para captura de pantalla")
    print("\n⚠️  Asegúrate de estar visible completo en la cámara")
    print()
    
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            # Dibujar landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                landmarks_buffer.append(results.pose_landmarks.landmark)
            
            # Clasificar
            if len(landmarks_buffer) >= WINDOW_SIZE:
                features = extract_features_from_window(list(landmarks_buffer))
                
                if features is not None:
                    features_scaled = scaler.transform([features])
                    proba = model.predict_proba(features_scaled)[0]
                    pred_idx = np.argmax(proba)
                    pred_label = metadata['class_names'][pred_idx]
                    confidence = proba[pred_idx]
                    
                    # Mostrar resultado principal
                    color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
                    cv2.putText(
                        frame,
                        f"{pred_label}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        color,
                        3
                    )
                    cv2.putText(
                        frame,
                        f"{confidence:.1%}",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2
                    )
                    
                    # Mostrar todas las probabilidades
                    y_offset = 120
                    sorted_indices = np.argsort(proba)[::-1]
                    for idx in sorted_indices:
                        activity = metadata['class_names'][idx]
                        prob = proba[idx]
                        
                        # Marcador visual
                        if prob > 0.3:
                            marker = "✅"
                            text_color = (0, 255, 0)
                        elif prob > 0.2:
                            marker = "⚠️ "
                            text_color = (0, 165, 255)
                        else:
                            marker = ""
                            text_color = (200, 200, 200)
                        
                        text = f"{marker} {activity}: {prob:.1%}"
                        cv2.putText(
                            frame,
                            text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            text_color,
                            1
                        )
                        y_offset += 25
            
            # Info del buffer
            cv2.putText(
                frame,
                f"Buffer: {len(landmarks_buffer)}/{WINDOW_SIZE}",
                (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            
            # Info del modelo
            cv2.putText(
                frame,
                f"Modelo: XGBoost (F1={metadata['xgb_f1_macro']:.3f})",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            
            cv2.imshow('Clasificación Real-time - Modelo Re-entrenado', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_path = Path('screenshot_realtime.png')
                cv2.imwrite(str(screenshot_path), frame)
                print(f"📸 Screenshot guardada: {screenshot_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Aplicación cerrada")


if __name__ == "__main__":
    main()
