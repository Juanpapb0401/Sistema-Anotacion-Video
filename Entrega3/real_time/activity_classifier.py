"""
Clasificador de actividades en tiempo real
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import deque


class ActivityClassifier:
    """Clasificador de actividades humanas en tiempo real"""
    
    def __init__(self, model_path: str, scaler_path: str, label_encoder_path: str):
        """
        Inicializa el clasificador
        
        Args:
            model_path: Ruta al modelo entrenado (.pkl)
            scaler_path: Ruta al scaler (.pkl)
            label_encoder_path: Ruta al label encoder (.pkl)
        """
        print("📂 Cargando modelo y preprocesadores...")
        
        # Cargar modelo
        self.model = joblib.load(model_path)
        print(f"   ✅ Modelo cargado: {Path(model_path).name}")
        
        # Cargar scaler
        self.scaler = joblib.load(scaler_path)
        print(f"   ✅ Scaler cargado")
        
        # Cargar label encoder
        self.label_encoder = joblib.load(label_encoder_path)
        print(f"   ✅ Label encoder cargado")
        
        # Cargar mapeo de nombres reales de actividades
        self.activity_names = self._load_activity_names()
        print(f"   📋 Clases: {list(self.activity_names.values())}")
        
        # Cargar información de features esperadas
        self.feature_names = self._load_feature_names()
        
        # Buffer para suavizar predicciones (ventana deslizante)
        # Aumentado a 10 frames para dar más estabilidad en acciones cortas
        self.prediction_buffer = deque(maxlen=10)  # Últimas 10 predicciones
        self.confidence_buffer = deque(maxlen=10)
        
        # Estadísticas - usar índices numéricos para evitar problemas
        self.total_predictions = 0
        self.class_counts = {i: 0 for i in range(len(self.label_encoder.classes_))}
    
    def _load_activity_names(self) -> dict:
        """Carga el mapeo de labels a nombres de actividades reales
        
        El label_encoder puede tener solo números si train.csv fue guardado
        con labels ya encodificados. Cargamos los nombres reales desde
        preparation_info.json que tiene el mapeo correcto.
        
        Returns:
            dict: Mapeo {índice_numérico: nombre_actividad}
        """
        try:
            data_path = Path(__file__).parent.parent.parent / "Entrega2" / "data"
            prep_info_path = data_path / "preparation_info.json"
            
            if prep_info_path.exists():
                import json
                with open(prep_info_path, 'r', encoding='utf-8') as f:
                    prep_info = json.load(f)
                
                # class_names está en orden alfabético (mismo que LabelEncoder)
                class_names = prep_info.get('class_names', [])
                
                if class_names:
                    activity_dict = {i: name for i, name in enumerate(class_names)}
                    
                    print(f"   📋 Mapeo de clases cargado desde preparation_info.json:")
                    for label, name in activity_dict.items():
                        print(f"      {label} → {name}")
                    
                    return activity_dict
            
            # Fallback: si label_encoder tiene strings, usarlos
            if isinstance(self.label_encoder.classes_[0], str):
                activity_dict = {i: cls for i, cls in enumerate(self.label_encoder.classes_)}
                print(f"   📋 Usando nombres de label_encoder.classes_")
                return activity_dict
            
            # Último fallback: nombres genéricos
            print("   ⚠️  No se encontraron nombres de clases, usando genéricos")
            return {i: f"Clase_{i}" for i in range(len(self.label_encoder.classes_))}
            
        except Exception as e:
            print(f"   ⚠️  Error al cargar nombres: {e}")
            # Fallback: usar números
            return {i: f"Clase_{i}" for i in range(len(self.label_encoder.classes_))}
                
        except Exception as e:
            print(f"   ⚠️  Error cargando nombres de actividades: {e}")
            # Fallback: usar los nombres del label_encoder como strings
            return {cls: f"Clase_{cls}" for cls in self.label_encoder.classes_}
        
    def _load_feature_names(self) -> list:
        """Carga los nombres de features desde preparation_info.json"""
        try:
            models_path = Path(__file__).parent.parent.parent / "Entrega2" / "data"
            import json
            with open(models_path / "preparation_info.json", 'r') as f:
                info = json.load(f)
                return info['feature_names']
        except Exception as e:
            print(f"⚠️  No se pudieron cargar feature_names: {e}")
            return None
    
    def prepare_features(self, features_dict: Dict[str, float]) -> np.ndarray:
        """
        Prepara las features para clasificación
        
        Args:
            features_dict: Diccionario con features extraídas del video
            
        Returns:
            Array numpy con features preparadas y normalizadas
        """
        if self.feature_names is None:
            # Si no tenemos los nombres, usar todas las features disponibles
            feature_values = list(features_dict.values())
        else:
            # Asegurar que las features estén en el orden correcto
            feature_values = []
            for fname in self.feature_names:
                if fname in features_dict:
                    feature_values.append(features_dict[fname])
                else:
                    # Si falta una feature, usar 0
                    feature_values.append(0.0)
        
        # Convertir a array y reshape para un sample
        X = np.array(feature_values).reshape(1, -1)
        
        # Normalizar con el scaler
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, features_dict: Dict[str, float], 
                use_smoothing: bool = True) -> Tuple[str, float, Dict[str, float]]:
        """
        Predice la actividad
        
        Args:
            features_dict: Diccionario con features extraídas
            use_smoothing: Si True, usa suavizado temporal de predicciones
            
        Returns:
            Tupla con (clase_predicha, confianza, probabilidades_todas_clases)
        """
        # Preparar features
        X = self.prepare_features(features_dict)
        
        # Predecir
        prediction = self.model.predict(X)[0]
        
        # Obtener probabilidades (si el modelo las soporta)
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[0]
        else:
            # Para modelos sin predict_proba, usar confianza binaria
            probabilities = np.zeros(len(self.label_encoder.classes_))
            probabilities[prediction] = 1.0
        
        # Agregar a buffer para suavizado
        if use_smoothing:
            self.prediction_buffer.append(prediction)
            self.confidence_buffer.append(probabilities[prediction])
            
            # Usar la predicción más frecuente en el buffer
            from collections import Counter
            prediction_counts = Counter(self.prediction_buffer)
            prediction = prediction_counts.most_common(1)[0][0]
            
            # Promedio de confianza
            confidence = np.mean(self.confidence_buffer)
        else:
            confidence = probabilities[prediction]
        
        # Obtener el nombre de la actividad directamente del número predicho
        # prediction es un número (0-4) que corresponde al índice en label_encoder.classes_
        class_name = self.activity_names.get(prediction, f"Clase_{prediction}")
        
        # DEBUG: Imprimir cada 30 frames
        if self.total_predictions % 30 == 0:
            print(f"\n🔍 DEBUG Prediction #{self.total_predictions}:")
            print(f"   Número predicho: {prediction}")
            print(f"   Clase mapeada: {class_name}")
            print(f"   label_encoder.classes_[{prediction}] = {self.label_encoder.classes_[prediction]}")
        
        # Crear diccionario de probabilidades por clase (con nombres reales)
        proba_dict = {}
        for i, prob in enumerate(probabilities):
            activity_name = self.activity_names.get(i, f"Clase_{i}")
            proba_dict[activity_name] = float(prob)
        
        # Actualizar estadísticas (usar índice numérico)
        self.total_predictions += 1
        self.class_counts[prediction] += 1
        
        return class_name, confidence, proba_dict
    
    def predict_with_metadata(self, features_dict: Dict[str, float]) -> Dict:
        """
        Predice y retorna información completa
        
        Args:
            features_dict: Features extraídas del frame
            
        Returns:
            Diccionario completo con predicción y metadata
        """
        class_name, confidence, probabilities = self.predict(features_dict)
        
        # Ordenar probabilidades de mayor a menor
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'class': class_name,
            'confidence': confidence,
            'probabilities': probabilities,
            'top_3_predictions': sorted_probs[:3],
            'total_predictions': self.total_predictions,
            'class_distribution': self.class_counts.copy()
        }
    
    def reset_buffers(self):
        """Reinicia los buffers de suavizado"""
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
    
    def get_statistics(self) -> Dict:
        """
        Obtiene estadísticas de las predicciones
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'total_predictions': self.total_predictions,
            'class_distribution': self.class_counts.copy(),
            'class_percentages': {}
        }
        
        if self.total_predictions > 0:
            for cls, count in self.class_counts.items():
                percentage = (count / self.total_predictions) * 100
                stats['class_percentages'][cls] = percentage
        
        return stats


def test_classifier():
    """Función de prueba para el clasificador"""
    print("🧪 Probando ActivityClassifier...")
    
    # Rutas a los archivos necesarios
    base_path = Path(__file__).parent.parent.parent / "Entrega2"
    model_path = base_path / "models" / "best_model.pkl"
    scaler_path = base_path / "models" / "scaler.pkl"
    label_encoder_path = base_path / "models" / "label_encoder.pkl"
    
    # Verificar que existan los archivos
    if not model_path.exists():
        print(f"❌ No se encontró el modelo: {model_path}")
        print("   Asegúrate de haber ejecutado 04_model_training.py primero")
        return
    
    # Inicializar clasificador
    classifier = ActivityClassifier(
        str(model_path),
        str(scaler_path),
        str(label_encoder_path)
    )
    
    print("\n✅ Clasificador inicializado correctamente")
    print(f"   Clases disponibles: {classifier.label_encoder.classes_.tolist()}")
    
    # Crear features de ejemplo (dummy)
    print("\n📊 Probando con features de ejemplo...")
    
    # Generar features aleatorias
    dummy_features = {}
    
    # Landmarks (33 puntos x 4 coordenadas)
    for i in range(33):
        dummy_features[f'landmark_{i}_x'] = np.random.random()
        dummy_features[f'landmark_{i}_y'] = np.random.random()
        dummy_features[f'landmark_{i}_z'] = np.random.random()
        dummy_features[f'landmark_{i}_visibility'] = 0.9
    
    # Ángulos
    dummy_features['left_elbow_angle'] = 45.0
    dummy_features['right_elbow_angle'] = 45.0
    dummy_features['left_knee_angle'] = 90.0
    dummy_features['right_knee_angle'] = 90.0
    
    # Hacer predicción
    result = classifier.predict_with_metadata(dummy_features)
    
    print(f"\n🎯 Predicción:")
    print(f"   Clase: {result['class']}")
    print(f"   Confianza: {result['confidence']:.2%}")
    print(f"\n   Top 3 predicciones:")
    for cls, prob in result['top_3_predictions']:
        print(f"   - {cls}: {prob:.2%}")
    
    print("\n✅ Test completado")


if __name__ == "__main__":
    test_classifier()
