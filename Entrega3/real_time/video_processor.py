"""
Procesador de video con MediaPipe para extracción de landmarks en tiempo real
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict


class VideoProcessor:
    """Clase para procesar video y extraer landmarks de pose en tiempo real"""
    
    def __init__(self):
        """Inicializa MediaPipe Pose"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Inicializar Pose con configuración optimizada para tiempo real
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1  # 0=lite, 1=full, 2=heavy
        )
        
        # Buffer para calcular velocidades y aceleraciones
        self.landmarks_history = []  # Últimos 3 frames para calcular velocidades
        self.angles_history = []  # Últimos 3 frames de ángulos
        self.distances_history = []  # Últimos 3 frames de distancias
        self.frame_count = 0
        self.max_history = 3  # Mantener últimos 3 frames
        
    def extract_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Extrae landmarks de un frame
        
        Args:
            frame: Frame de video (BGR)
            
        Returns:
            Diccionario con landmarks y frame procesado, o None si no se detecta persona
        """
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar con MediaPipe
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Extraer coordenadas normalizadas
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        self.frame_count += 1
        
        return {
            'landmarks': landmarks,
            'results': results,
            'frame_count': self.frame_count
        }
    
    def draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Dibuja los landmarks sobre el frame
        
        Args:
            frame: Frame original
            results: Resultados de MediaPipe
            
        Returns:
            Frame con landmarks dibujados
        """
        if results and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame
    
    def compute_angles(self, landmarks_array: np.ndarray) -> Dict[str, float]:
        """
        Calcula ángulos entre articulaciones principales
        
        Args:
            landmarks_array: Array con las 33 landmarks (x, y, z, visibility)
            
        Returns:
            Diccionario con ángulos calculados
        """
        # Reshape a (33, 4)
        landmarks = landmarks_array.reshape(33, 4)
        
        def calculate_angle(a, b, c):
            """Calcula ángulo entre tres puntos"""
            a = np.array([landmarks[a, 0], landmarks[a, 1]])
            b = np.array([landmarks[b, 0], landmarks[b, 1]])
            c = np.array([landmarks[c, 0], landmarks[c, 1]])
            
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                      np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            
            if angle > 180.0:
                angle = 360 - angle
            
            return angle
        
        # Índices de landmarks clave (según MediaPipe)
        # 11: left_shoulder, 13: left_elbow, 15: left_wrist
        # 12: right_shoulder, 14: right_elbow, 16: right_wrist
        # 23: left_hip, 25: left_knee, 27: left_ankle
        # 24: right_hip, 26: right_knee, 28: right_ankle
        
        angles = {}
        
        try:
            # Ángulos de brazos
            angles['left_elbow_angle'] = calculate_angle(11, 13, 15)
            angles['right_elbow_angle'] = calculate_angle(12, 14, 16)
            
            # Ángulos de piernas
            angles['left_knee_angle'] = calculate_angle(23, 25, 27)
            angles['right_knee_angle'] = calculate_angle(24, 26, 28)
            
            # Ángulos de cadera
            angles['left_hip_angle'] = calculate_angle(11, 23, 25)
            angles['right_hip_angle'] = calculate_angle(12, 24, 26)
            
            # Inclinación del tronco (diferencia entre hombros y caderas)
            left_shoulder = landmarks[11, :2]
            right_shoulder = landmarks[12, :2]
            left_hip = landmarks[23, :2]
            right_hip = landmarks[24, :2]
            
            shoulder_midpoint = (left_shoulder + right_shoulder) / 2
            hip_midpoint = (left_hip + right_hip) / 2
            
            trunk_angle = np.arctan2(
                shoulder_midpoint[1] - hip_midpoint[1],
                shoulder_midpoint[0] - hip_midpoint[0]
            ) * 180 / np.pi
            
            angles['trunk_inclination'] = abs(trunk_angle - 90)  # Desviación de vertical
            
        except Exception as e:
            print(f"Error calculando ángulos: {e}")
            # Retornar ángulos por defecto
            return {
                'left_elbow_angle': 0,
                'right_elbow_angle': 0,
                'left_knee_angle': 0,
                'right_knee_angle': 0,
                'left_hip_angle': 0,
                'right_hip_angle': 0,
                'trunk_inclination': 0
            }
        
        return angles
    
    def compute_distances(self, landmarks_array: np.ndarray) -> Dict[str, float]:
        """
        Calcula distancias entre puntos clave
        
        Args:
            landmarks_array: Array con las 33 landmarks
            
        Returns:
            Diccionario con distancias calculadas
        """
        landmarks = landmarks_array.reshape(33, 4)
        
        def euclidean_distance(idx1, idx2):
            """Calcula distancia euclidiana entre dos landmarks"""
            p1 = landmarks[idx1, :2]
            p2 = landmarks[idx2, :2]
            return np.linalg.norm(p1 - p2)
        
        distances = {}
        
        try:
            # Distancias clave
            distances['shoulder_width'] = euclidean_distance(11, 12)
            distances['hip_width'] = euclidean_distance(23, 24)
            distances['torso_length'] = euclidean_distance(11, 23)  # shoulder to hip
            distances['left_arm_length'] = euclidean_distance(11, 15)  # shoulder to wrist
            distances['right_arm_length'] = euclidean_distance(12, 16)
            distances['left_leg_length'] = euclidean_distance(23, 27)  # hip to ankle
            distances['right_leg_length'] = euclidean_distance(24, 28)
            
        except Exception as e:
            print(f"Error calculando distancias: {e}")
            return {
                'shoulder_width': 0,
                'hip_width': 0,
                'torso_length': 0,
                'left_arm_length': 0,
                'right_arm_length': 0,
                'left_leg_length': 0,
                'right_leg_length': 0
            }
        
        return distances
    
    def compute_velocities(self, current_landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calcula velocidades de ángulos y distancias (NO de landmarks individuales)
        
        Args:
            current_landmarks: Landmarks del frame actual
            
        Returns:
            Diccionario con velocidades de ángulos y distancias
        """
        # Calcular ángulos y distancias actuales
        current_angles = self.compute_angles(current_landmarks)
        current_distances = self.compute_distances(current_landmarks)
        
        # Agregar a historial
        self.angles_history.append(current_angles)
        self.distances_history.append(current_distances)
        
        # Mantener solo los últimos N frames
        if len(self.angles_history) > self.max_history:
            self.angles_history.pop(0)
        if len(self.distances_history) > self.max_history:
            self.distances_history.pop(0)
        
        velocities = {}
        
        # Si no hay frames previos, retornar ceros
        if len(self.angles_history) < 2:
            # Velocidades de ángulos
            velocities['left_knee_angle_velocity'] = 0.0
            velocities['right_knee_angle_velocity'] = 0.0
            velocities['left_elbow_angle_velocity'] = 0.0
            velocities['right_elbow_angle_velocity'] = 0.0
            # Velocidades de distancias
            velocities['left_wrist_dist_velocity'] = 0.0
        else:
            # Calcular velocidades (diferencia entre frame actual y anterior)
            prev_angles = self.angles_history[-2]
            curr_angles = self.angles_history[-1]
            
            prev_distances = self.distances_history[-2]
            curr_distances = self.distances_history[-1]
            
            # Velocidades de ángulos
            velocities['left_knee_angle_velocity'] = curr_angles['left_knee_angle'] - prev_angles['left_knee_angle']
            velocities['right_knee_angle_velocity'] = curr_angles['right_knee_angle'] - prev_angles['right_knee_angle']
            velocities['left_elbow_angle_velocity'] = curr_angles['left_elbow_angle'] - prev_angles['left_elbow_angle']
            velocities['right_elbow_angle_velocity'] = curr_angles['right_elbow_angle'] - prev_angles['right_elbow_angle']
            
            # Velocidad de distancia (usando left_arm_length como proxy de left_wrist_dist)
            velocities['left_wrist_dist_velocity'] = curr_distances['left_arm_length'] - prev_distances['left_arm_length']
        
        return velocities
    
    def compute_accelerations(self) -> Dict[str, float]:
        """
        Calcula aceleraciones (cambio de velocidad)
        
        Returns:
            Diccionario con aceleraciones
        """
        accelerations = {}
        
        # Si no hay suficientes frames, retornar ceros
        if len(self.angles_history) < 3:
            accelerations['left_knee_angle_velocity_accel'] = 0.0
            accelerations['right_knee_angle_velocity_accel'] = 0.0
            accelerations['left_elbow_angle_velocity_accel'] = 0.0
            accelerations['right_elbow_angle_velocity_accel'] = 0.0
            accelerations['left_wrist_dist_velocity_accel'] = 0.0
        else:
            # Calcular velocidades en t-2, t-1, t
            angles_t_2 = self.angles_history[-3]
            angles_t_1 = self.angles_history[-2]
            angles_t = self.angles_history[-1]
            
            distances_t_2 = self.distances_history[-3]
            distances_t_1 = self.distances_history[-2]
            distances_t = self.distances_history[-1]
            
            # Velocidades en t-1 y t
            vel_t_1 = {
                'left_knee': angles_t_1['left_knee_angle'] - angles_t_2['left_knee_angle'],
                'right_knee': angles_t_1['right_knee_angle'] - angles_t_2['right_knee_angle'],
                'left_elbow': angles_t_1['left_elbow_angle'] - angles_t_2['left_elbow_angle'],
                'right_elbow': angles_t_1['right_elbow_angle'] - angles_t_2['right_elbow_angle'],
                'left_wrist_dist': distances_t_1['left_arm_length'] - distances_t_2['left_arm_length']
            }
            
            vel_t = {
                'left_knee': angles_t['left_knee_angle'] - angles_t_1['left_knee_angle'],
                'right_knee': angles_t['right_knee_angle'] - angles_t_1['right_knee_angle'],
                'left_elbow': angles_t['left_elbow_angle'] - angles_t_1['left_elbow_angle'],
                'right_elbow': angles_t['right_elbow_angle'] - angles_t_1['right_elbow_angle'],
                'left_wrist_dist': distances_t['left_arm_length'] - distances_t_1['left_arm_length']
            }
            
            # Aceleraciones (cambio de velocidad)
            accelerations['left_knee_angle_velocity_accel'] = vel_t['left_knee'] - vel_t_1['left_knee']
            accelerations['right_knee_angle_velocity_accel'] = vel_t['right_knee'] - vel_t_1['right_knee']
            accelerations['left_elbow_angle_velocity_accel'] = vel_t['left_elbow'] - vel_t_1['left_elbow']
            accelerations['right_elbow_angle_velocity_accel'] = vel_t['right_elbow'] - vel_t_1['right_elbow']
            accelerations['left_wrist_dist_velocity_accel'] = vel_t['left_wrist_dist'] - vel_t_1['left_wrist_dist']
        
        return accelerations
    
    def extract_features_from_landmarks(self, landmarks_array: np.ndarray) -> Dict[str, float]:
        """
        Extrae todas las features necesarias para clasificación
        
        Args:
            landmarks_array: Array con landmarks (132 valores: 33 landmarks x 4 coords)
            
        Returns:
            Diccionario con todas las features
        """
        features = {}
        
        # 1. Landmarks base (x, y, z, visibility para cada uno de los 33 puntos)
        # IMPORTANTE: Usar nombres compatibles con el entrenamiento (x_0, y_0, z_0, v_0)
        landmarks_reshaped = landmarks_array.reshape(33, 4)
        for i in range(33):
            features[f'x_{i}'] = landmarks_reshaped[i, 0]
            features[f'y_{i}'] = landmarks_reshaped[i, 1]
            features[f'z_{i}'] = landmarks_reshaped[i, 2]
            features[f'v_{i}'] = landmarks_reshaped[i, 3]
        
        # 2. Ángulos
        angles = self.compute_angles(landmarks_array)
        features.update(angles)
        
        # 3. Distancias - usar solo left_wrist_dist (left_arm_length como proxy)
        distances = self.compute_distances(landmarks_array)
        # El modelo espera 'left_wrist_dist' pero calculamos 'left_arm_length'
        features['left_wrist_dist'] = distances['left_arm_length']
        
        # 4. Velocidades (incluye cálculo de ángulos y distancias para historial)
        velocities = self.compute_velocities(landmarks_array)
        features.update(velocities)
        
        # 5. Aceleraciones
        accelerations = self.compute_accelerations()
        features.update(accelerations)
        
        return features
    
    def release(self):
        """Libera recursos de MediaPipe"""
        self.pose.close()


def test_video_processor():
    """Función de prueba para el procesador de video"""
    print("🎥 Probando VideoProcessor con cámara web...")
    
    processor = VideoProcessor()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" No se pudo abrir la cámara")
        return
    
    print(" Cámara abierta. Presiona 'q' para salir.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(" No se pudo leer el frame")
            break
        
        # Extraer landmarks
        result = processor.extract_landmarks(frame)
        
        if result:
            # Dibujar landmarks
            frame = processor.draw_landmarks(frame, result['results'])
            
            # Calcular features
            landmarks_array = np.array(result['landmarks'])
            features = processor.extract_features_from_landmarks(landmarks_array)
            
            # Mostrar algunas features
            angles = processor.compute_angles(landmarks_array)
            
            # Agregar texto al frame
            y_offset = 30
            for key, value in list(angles.items())[:4]:  # Mostrar solo 4 ángulos
                text = f"{key}: {value:.1f}°"
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 30
        else:
            cv2.putText(frame, "No person detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Mostrar frame
        cv2.imshow('Video Processor Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    processor.release()
    
    print(" Test completado")


if __name__ == "__main__":
    test_video_processor()
