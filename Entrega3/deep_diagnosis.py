"""
Diagnóstico PROFUNDO del sistema de clasificación
Compara features en tiempo real con el dataset de entrenamiento
"""
import cv2
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import joblib
from collections import Counter, defaultdict

# Agregar paths
sys.path.append(str(Path(__file__).parent))
from real_time.video_processor import VideoProcessor
from real_time.activity_classifier import ActivityClassifier

def load_training_data():
    """Carga estadísticas del dataset de entrenamiento"""
    base_path = Path(__file__).parent.parent
    train_path = base_path / "Entrega2" / "data" / "train.csv"
    
    if not train_path.exists():
        print("⚠️ No se encontró train.csv")
        return None
    
    df = pd.read_csv(train_path)
    print(f"✅ Dataset de entrenamiento cargado: {len(df)} muestras")
    
    # Separar features y labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    return X, y, df

def analyze_training_patterns():
    """Analiza patrones del dataset de entrenamiento por clase"""
    X, y, df = load_training_data()
    if X is None:
        return None
    
    stats = {}
    
    # Obtener clases únicas (ordenadas alfabéticamente como LabelEncoder)
    classes = sorted(y.unique())
    
    print("\n" + "=" * 80)
    print("📊 ESTADÍSTICAS POR CLASE EN TRAINING DATA")
    print("=" * 80)
    
    for cls in classes:
        class_data = X[y == cls]
        
        # Features clave para diferenciar actividades
        key_features = [
            'left_knee_angle', 'right_knee_angle',  # Sentarse/pararse
            'left_elbow_angle', 'right_elbow_angle',  # Movimiento brazos
            'left_knee_angle_velocity', 'right_knee_angle_velocity',  # Velocidad rodillas
            'left_wrist_dist_velocity',  # Velocidad manos
            'x_0', 'y_0', 'z_0',  # Posición nariz (landmark 0)
        ]
        
        stats[cls] = {}
        
        print(f"\n🏷️  {cls}:")
        print(f"   Muestras: {len(class_data)}")
        
        for feature in key_features:
            if feature in class_data.columns:
                mean_val = class_data[feature].mean()
                std_val = class_data[feature].std()
                stats[cls][feature] = {'mean': mean_val, 'std': std_val}
                print(f"   {feature:30s} → mean={mean_val:8.4f}, std={std_val:8.4f}")
    
    return stats, classes

def diagnose_realtime():
    """Diagnóstico en tiempo real con comparación vs training data"""
    
    print("\n" + "=" * 80)
    print("🔬 DIAGNÓSTICO PROFUNDO - COMPARACIÓN CON TRAINING DATA")
    print("=" * 80)
    
    # Cargar estadísticas de entrenamiento
    train_stats, train_classes = analyze_training_patterns()
    
    if train_stats is None:
        print("❌ No se pudo cargar training data")
        return
    
    print("\n" + "=" * 80)
    print("🎥 INICIANDO CAPTURA EN TIEMPO REAL")
    print("=" * 80)
    print("\nInstrucciones:")
    print("  1 - Capturar mientras caminas HACIA la cámara (3 seg)")
    print("  2 - Capturar mientras caminas DE REGRESO (3 seg)")
    print("  3 - Capturar mientras GIRAS (3 seg)")
    print("  4 - Capturar mientras te SIENTAS (3 seg)")
    print("  5 - Capturar mientras te PARAS (3 seg)")
    print("  'q' - Salir")
    print()
    
    # Inicializar componentes
    processor = VideoProcessor()
    classifier = ActivityClassifier()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return
    
    # Almacenar capturas por actividad
    captures = {
        '1': {'name': 'caminar_hacia_camara', 'frames': []},
        '2': {'name': 'caminar_de_regreso', 'frames': []},
        '3': {'name': 'girar', 'frames': []},
        '4': {'name': 'sentarse', 'frames': []},
        '5': {'name': 'ponerse_de_pie', 'frames': []}
    }
    
    current_capture = None
    capture_start_time = None
    capture_duration = 3.0  # segundos
    
    print("✅ Cámara iniciada. Presiona 1-5 para capturar cada actividad...")
    print()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar frame
        results = processor.process_frame(frame)
        
        if results and results.pose_landmarks:
            # Extraer features
            features = processor.extract_features_from_landmarks(results.pose_landmarks)
            
            # Predecir
            activity, confidence, probabilities = classifier.predict(features)
            
            # Si estamos capturando
            if current_capture:
                elapsed = time.time() - capture_start_time
                if elapsed < capture_duration:
                    captures[current_capture]['frames'].append({
                        'features': features.copy(),
                        'prediction': activity,
                        'confidence': confidence,
                        'probabilities': probabilities.copy()
                    })
                else:
                    # Terminar captura
                    print(f"\n✅ Captura completada: {captures[current_capture]['name']}")
                    analyze_capture(captures[current_capture], train_stats, train_classes)
                    current_capture = None
            
            # Dibujar skeleton
            processor.draw_skeleton(frame, results.pose_landmarks)
            
            # Mostrar info
            y_pos = 30
            
            if current_capture:
                elapsed = time.time() - capture_start_time
                remaining = capture_duration - elapsed
                cv2.putText(frame, f"CAPTURANDO: {captures[current_capture]['name']}", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_pos += 30
                cv2.putText(frame, f"Tiempo restante: {remaining:.1f}s", 
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_pos += 30
            
            cv2.putText(frame, f"Prediccion: {activity}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
            cv2.putText(frame, f"Confianza: {confidence:.3f}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 40
            
            # Mostrar probabilidades
            for act_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                color = (0, 255, 0) if prob > 0.3 else (200, 200, 200)
                cv2.putText(frame, f"{act_name}: {prob:.3f}", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 22
        
        cv2.putText(frame, "1:Hacia | 2:Regreso | 3:Girar | 4:Sentar | 5:Parar | Q:Salir", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("Diagnostico Profundo", frame)
        
        # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif chr(key) in captures.keys() and current_capture is None:
            current_capture = chr(key)
            capture_start_time = time.time()
            print(f"\n🎬 Iniciando captura: {captures[current_capture]['name']}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Resumen final
    print("\n" + "=" * 80)
    print("📈 RESUMEN FINAL")
    print("=" * 80)
    
    for key, data in captures.items():
        if data['frames']:
            print(f"\n✅ {data['name']}: {len(data['frames'])} frames capturados")
        else:
            print(f"\n⚠️  {data['name']}: No capturado")

def analyze_capture(capture_data, train_stats, train_classes):
    """Analiza una captura y la compara con training data"""
    
    frames = capture_data['frames']
    expected_class = capture_data['name']
    
    if not frames:
        return
    
    print(f"\n{'=' * 80}")
    print(f"🔍 ANÁLISIS: {expected_class}")
    print(f"{'=' * 80}")
    print(f"Frames capturados: {len(frames)}")
    
    # 1. Distribución de predicciones
    predictions = [f['prediction'] for f in frames]
    pred_counts = Counter(predictions)
    
    print(f"\n📊 Predicciones del modelo:")
    for pred, count in pred_counts.most_common():
        pct = (count / len(frames)) * 100
        marker = "✅" if pred == expected_class else "❌"
        print(f"   {marker} {pred}: {count} frames ({pct:.1f}%)")
    
    # 2. Probabilidades promedio
    all_probs = defaultdict(list)
    for f in frames:
        for cls, prob in f['probabilities'].items():
            all_probs[cls].append(prob)
    
    print(f"\n📈 Probabilidades promedio:")
    for cls in sorted(all_probs.keys()):
        avg_prob = np.mean(all_probs[cls])
        marker = "🎯" if cls == expected_class else "  "
        print(f"   {marker} {cls}: {avg_prob:.3f}")
    
    # 3. Comparación de features con training data
    print(f"\n🔬 Comparación de features con training data:")
    
    key_features = [
        'left_knee_angle', 'right_knee_angle',
        'left_knee_angle_velocity', 'right_knee_angle_velocity',
        'left_wrist_dist_velocity'
    ]
    
    for feature in key_features:
        # Valor promedio en esta captura
        feature_values = [f['features'].get(feature, 0) for f in frames if feature in f['features']]
        if not feature_values:
            continue
        
        capture_mean = np.mean(feature_values)
        
        # Valor esperado en training data
        if expected_class in train_stats and feature in train_stats[expected_class]:
            train_mean = train_stats[expected_class][feature]['mean']
            train_std = train_stats[expected_class][feature]['std']
            
            diff = abs(capture_mean - train_mean)
            z_score = diff / train_std if train_std > 0 else 0
            
            status = "✅" if z_score < 2 else "⚠️" if z_score < 3 else "❌"
            
            print(f"   {status} {feature:30s}")
            print(f"      Captura:  {capture_mean:8.4f}")
            print(f"      Training: {train_mean:8.4f} ± {train_std:8.4f}")
            print(f"      Z-score:  {z_score:.2f} {'(NORMAL)' if z_score < 2 else '(ANORMAL)'}")
    
    # 4. ¿A qué clase se parece más?
    print(f"\n🎯 Similitud con clases de training:")
    similarities = {}
    
    for cls in train_classes:
        total_distance = 0
        count = 0
        
        for feature in key_features:
            feature_values = [f['features'].get(feature, 0) for f in frames if feature in f['features']]
            if not feature_values or cls not in train_stats or feature not in train_stats[cls]:
                continue
            
            capture_mean = np.mean(feature_values)
            train_mean = train_stats[cls][feature]['mean']
            train_std = train_stats[cls][feature]['std']
            
            if train_std > 0:
                z_score = abs(capture_mean - train_mean) / train_std
                total_distance += z_score
                count += 1
        
        if count > 0:
            avg_distance = total_distance / count
            similarities[cls] = avg_distance
    
    for cls, dist in sorted(similarities.items(), key=lambda x: x[1]):
        marker = "🎯" if cls == expected_class else "  "
        print(f"   {marker} {cls}: distancia={dist:.2f}")
    
    print()

if __name__ == "__main__":
    diagnose_realtime()
