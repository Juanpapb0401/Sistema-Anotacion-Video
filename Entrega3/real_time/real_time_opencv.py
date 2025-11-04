"""
Aplicación de clasificación en tiempo real - Versión con OpenCV nativo
Similar al notebook de ejemplo pero integrado con nuestros modelos
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import time

# Añadir el directorio al path
sys.path.insert(0, str(Path(__file__).parent))

from video_processor import VideoProcessor
from activity_classifier import ActivityClassifier


def main():
    """Función principal - loop continuo como en el notebook"""
    
    print("🎥 Sistema de Clasificación de Actividades en Tiempo Real")
    print("=" * 60)
    print()
    
    # Configurar paths de modelos
    base_path = Path(__file__).parent.parent.parent
    model_path = base_path / 'Entrega2' / 'models' / 'best_model.pkl'
    scaler_path = base_path / 'Entrega2' / 'models' / 'scaler.pkl'
    label_encoder_path = base_path / 'Entrega2' / 'models' / 'label_encoder.pkl'
    
    # Verificar que existen los modelos
    if not all([model_path.exists(), scaler_path.exists(), label_encoder_path.exists()]):
        print(" Error: No se encontraron los modelos entrenados")
        print("   Ejecuta primero: cd Entrega2/notebooks && python 04_model_training_gridsearch.py")
        return
    
    print(" Cargando modelos...")
    
    # Inicializar procesador y clasificador
    video_processor = VideoProcessor()
    classifier = ActivityClassifier(
        str(model_path),
        str(scaler_path),
        str(label_encoder_path)
    )
    
    print(" Modelos cargados")
    print()
    
    # Configurar permisos de cámara en macOS
    import os
    os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'
    
    # Inicializar cámara (como en el notebook)
    print("📹 Abriendo cámara...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" No se pudo abrir la cámara")
        print()
        print("Solución para macOS:")
        print("1. Abre Preferencias del Sistema → Seguridad y Privacidad")
        print("2. Ve a Privacidad → Cámara")
        print("3. Habilita permisos para Terminal/Python")
        return
    
    print(" Cámara activa")
    print()
    print(" Instrucciones:")
    print("   - Realiza una actividad frente a la cámara")
    print("   - Verás la predicción en tiempo real")
    print("   - Presiona 'q' para salir")
    print()
    print("🎬 Iniciando análisis continuo...")
    print()
    
    # Configuración
    confidence_threshold = 0.8  # Aumentado para evitar predicciones inciertas
    frame_count = 0
    
    # Loop continuo (EXACTAMENTE como en el notebook)
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("️ Error al leer frame")
            break
        
        frame_count += 1
        
        # Crear copia para anotar (como en el notebook)
        annotated_frame = frame.copy()
        
        # Extraer landmarks
        result = video_processor.extract_landmarks(frame)
        
        if result:
            landmarks_array = np.array(result['landmarks'])
            
            # Dibujar landmarks
            annotated_frame = video_processor.draw_landmarks(annotated_frame, result['results'])
            
            # Extraer features
            features = video_processor.extract_features_from_landmarks(landmarks_array)
            
            # Clasificar SIN smoothing para ver predicciones crudas
            # Cambiar use_smoothing=False temporalmente para debug
            activity, confidence, probabilities = classifier.predict(features, use_smoothing=False)
            
            # Crear metadata (compatible con código anterior)
            prediction = {
                'class': activity,
                'confidence': confidence,
                'probabilities': probabilities
            }
            
            # MOSTRAR PROBABILIDADES EN ORDEN (mayor a menor)
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            
            # Mostrar en consola
            print(f"\r[Frame {frame_count}] ", end="")
            for act_name, prob in sorted_probs:
                # Convertir a string por si es número
                act_str = str(act_name)
                # Color en terminal: verde si >0.3, amarillo si >0.2, blanco si <0.2
                if prob > 0.3:
                    print(f" {act_str}: {prob:.2%}", end=" | ")
                elif prob > 0.2:
                    print(f"️  {act_str}: {prob:.2%}", end=" | ")
                else:
                    print(f"   {act_str}: {prob:.2%}", end=" | ")
            
            # Convertir activity a string para upper()
            activity_str = str(activity)
            print(f"→ {activity_str.upper()}", end="")
            
            # Agregar texto al frame (como en el notebook: cv2.putText)
            pred_label = str(prediction['class'])  # Convertir a string por si es número
            conf_text = f"{prediction['confidence']:.1%}"
            
            # Texto grande para la actividad (siempre mostrar, con color según confianza)
            color = (0, 255, 0) if prediction['confidence'] >= confidence_threshold else (0, 165, 255)
            cv2.putText(annotated_frame, f'Actividad: {pred_label}', 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Texto para la confianza
            cv2.putText(annotated_frame, f'Confianza: {conf_text}', 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Mostrar TODAS las probabilidades en el video (DEBUG)
            y_pos = 140
            cv2.putText(annotated_frame, 'Todas las probabilidades:', 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            y_pos += 25
            for activity, prob in sorted(prediction['probabilities'].items(), 
                                        key=lambda x: x[1], reverse=True):
                # Convertir a string por si es número
                activity_str = str(activity)
                # Color verde para la elegida, blanco para las demás
                text_color = (0, 255, 0) if str(activity) == pred_label else (200, 200, 200)
                cv2.putText(annotated_frame, f'{activity_str}: {prob:.1%}', 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                y_pos += 20
            
            # Contador de frames
            cv2.putText(annotated_frame, f'Frame: {frame_count}', 
                       (annotated_frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            # No se detectó persona
            cv2.putText(annotated_frame, 'No se detecto persona', 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Mostrar frame (como en el notebook: cv2.imshow)
        cv2.imshow('Clasificacion de Actividades en Tiempo Real', annotated_frame)
        
        # Esperar tecla (como en el notebook: cv2.waitKey)
        # Si presionas 'q', sale del loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print()
            print("🛑 Deteniendo análisis...")
            break
    
    # Limpiar (como en el notebook)
    cap.release()
    cv2.destroyAllWindows()
    video_processor.release()
    
    print()
    print(" Sesión finalizada")
    print(f" Total de frames procesados: {frame_count}")
    print()


if __name__ == "__main__":
    main()
