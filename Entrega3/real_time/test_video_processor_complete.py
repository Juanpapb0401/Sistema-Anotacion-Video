#!/usr/bin/env python3
"""
Script de prueba completo del VideoProcessor
Puedes ejecutar esto AHORA sin necesitar los modelos entrenados
"""

import cv2
import numpy as np
import time
from pathlib import Path
import sys

# Agregar el directorio al path
sys.path.append(str(Path(__file__).parent))

from video_processor import VideoProcessor


def main():
    print("=" * 80)
    print("🎥 PRUEBA COMPLETA DEL VIDEO PROCESSOR")
    print("=" * 80)
    print()
    print("Esta prueba NO requiere modelos entrenados")
    print("Solo necesita:")
    print("  - opencv-python")
    print("  - mediapipe")
    print("  - numpy")
    print()
    print("Controles:")
    print("  - Presiona 'q' para salir")
    print("  - Presiona 's' para tomar screenshot")
    print("  - Presiona 'r' para reiniciar estadísticas")
    print()
    print("=" * 80)
    print()
    
    # Inicializar VideoProcessor
    print("📦 Inicializando VideoProcessor...")
    processor = VideoProcessor()
    print(" VideoProcessor listo")
    print()
    
    # Abrir cámara
    print("📹 Abriendo cámara web...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" No se pudo abrir la cámara")
        print("   Verifica que:")
        print("   1. Tienes una cámara conectada")
        print("   2. No está siendo usada por otra aplicación")
        print("   3. Tienes permisos de cámara en tu sistema")
        return
    
    print(" Cámara abierta correctamente")
    print()
    print("▶️  Procesando video en vivo...")
    print()
    
    # Estadísticas
    frame_count = 0
    start_time = time.time()
    fps_values = []
    angles_history = {
        'left_knee': [],
        'right_knee': [],
        'trunk_inclination': []
    }
    
    last_fps_update = time.time()
    current_fps = 0
    
    while cap.isOpened():
        frame_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print(" Error al leer frame")
            break
        
        frame_count += 1
        
        # Extraer landmarks
        result = processor.extract_landmarks(frame)
        
        if result:
            landmarks_array = np.array(result['landmarks'])
            
            # Calcular features
            angles = processor.compute_angles(landmarks_array)
            distances = processor.compute_distances(landmarks_array)
            
            # Guardar en historial
            if 'left_knee_angle' in angles:
                angles_history['left_knee'].append(angles['left_knee_angle'])
            if 'right_knee_angle' in angles:
                angles_history['right_knee'].append(angles['right_knee_angle'])
            if 'trunk_inclination' in angles:
                angles_history['trunk_inclination'].append(angles['trunk_inclination'])
            
            # Mantener solo últimos 100 frames
            for key in angles_history:
                if len(angles_history[key]) > 100:
                    angles_history[key].pop(0)
            
            # Dibujar landmarks
            frame = processor.draw_landmarks(frame, result['results'])
            
            # Preparar overlay de información
            overlay = frame.copy()
            
            # Fondo semi-transparente para el texto
            cv2.rectangle(overlay, (10, 10), (400, 280), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Información en el frame
            y_offset = 40
            line_height = 25
            
            # FPS
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += line_height
            
            # Frame count
            cv2.putText(frame, f"Frames: {frame_count}", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height + 10
            
            # Ángulos principales
            cv2.putText(frame, "=== ANGULOS ===", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += line_height
            
            angles_to_show = [
                ('left_knee_angle', 'Rodilla Izq'),
                ('right_knee_angle', 'Rodilla Der'),
                ('left_hip_angle', 'Cadera Izq'),
                ('trunk_inclination', 'Inclinacion')
            ]
            
            for angle_key, angle_label in angles_to_show:
                if angle_key in angles:
                    value = angles[angle_key]
                    # Color según valor
                    if 80 <= value <= 100:  # Rango normal para rodillas
                        color = (0, 255, 0)  # Verde
                    elif value > 150:
                        color = (0, 165, 255)  # Naranja
                    else:
                        color = (255, 255, 255)  # Blanco
                    
                    text = f"{angle_label}: {value:.1f}"
                    cv2.putText(frame, text, (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += line_height - 5
            
            y_offset += 10
            
            # Distancias
            cv2.putText(frame, "=== DISTANCIAS ===", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += line_height
            
            if 'shoulder_width' in distances:
                cv2.putText(frame, f"Hombros: {distances['shoulder_width']:.3f}", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += line_height - 5
            
            if 'torso_length' in distances:
                cv2.putText(frame, f"Torso: {distances['torso_length']:.3f}", (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Estadísticas promedio (esquina superior derecha)
            if len(angles_history['left_knee']) > 0:
                avg_left_knee = np.mean(angles_history['left_knee'])
                avg_right_knee = np.mean(angles_history['right_knee'])
                avg_trunk = np.mean(angles_history['trunk_inclination'])
                
                # Fondo para estadísticas
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (frame.shape[1] - 310, 10), 
                             (frame.shape[1] - 10, 140), (0, 0, 0), -1)
                frame = cv2.addWeighted(frame, 0.7, overlay2, 0.3, 0)
                
                y_offset2 = 40
                cv2.putText(frame, "=== PROMEDIOS ===", 
                           (frame.shape[1] - 290, y_offset2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset2 += 30
                
                cv2.putText(frame, f"Rodilla Izq: {avg_left_knee:.1f}", 
                           (frame.shape[1] - 290, y_offset2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset2 += 25
                
                cv2.putText(frame, f"Rodilla Der: {avg_right_knee:.1f}", 
                           (frame.shape[1] - 290, y_offset2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset2 += 25
                
                cv2.putText(frame, f"Inclinacion: {avg_trunk:.1f}", 
                           (frame.shape[1] - 290, y_offset2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        else:
            # No se detectó persona
            cv2.putText(frame, "No se detecto persona", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Instrucciones en la parte inferior
        instructions = [
            "q: Salir | s: Screenshot | r: Reiniciar stats"
        ]
        
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (0, frame.shape[0] - 40), 
                     (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay3, 0.3, 0)
        
        cv2.putText(frame, instructions[0], (20, frame.shape[0] - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Calcular FPS
        frame_end = time.time()
        frame_time = frame_end - frame_start
        if frame_time > 0:
            fps_values.append(1.0 / frame_time)
            if len(fps_values) > 30:  # Mantener últimos 30 valores
                fps_values.pop(0)
        
        # Actualizar FPS cada 0.5 segundos
        if frame_end - last_fps_update > 0.5:
            if len(fps_values) > 0:
                current_fps = np.mean(fps_values)
            last_fps_update = frame_end
        
        # Mostrar frame
        cv2.imshow('Video Processor - Prueba Completa', frame)
        
        # Manejar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n⏹️  Deteniendo...")
            break
        elif key == ord('s'):
            # Tomar screenshot
            screenshot_name = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(screenshot_name, frame)
            print(f"📸 Screenshot guardado: {screenshot_name}")
        elif key == ord('r'):
            # Reiniciar estadísticas
            angles_history = {k: [] for k in angles_history}
            frame_count = 0
            fps_values = []
            start_time = time.time()
            print(" Estadísticas reiniciadas")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    processor.release()
    
    # Estadísticas finales
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    print()
    print("=" * 80)
    print(" ESTADÍSTICAS FINALES")
    print("=" * 80)
    print(f"Total de frames procesados: {frame_count}")
    print(f"Tiempo total: {total_time:.2f}s")
    print(f"FPS promedio: {avg_fps:.2f}")
    print()
    
    if len(angles_history['left_knee']) > 0:
        print("Promedios de ángulos:")
        print(f"  - Rodilla izquierda: {np.mean(angles_history['left_knee']):.2f}°")
        print(f"  - Rodilla derecha: {np.mean(angles_history['right_knee']):.2f}°")
        print(f"  - Inclinación tronco: {np.mean(angles_history['trunk_inclination']):.2f}°")
        print()
    
    print(" Prueba completada exitosamente")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n️  Interrumpido por el usuario")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
