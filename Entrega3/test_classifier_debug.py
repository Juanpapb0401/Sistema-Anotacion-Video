"""
Script de diagnóstico para verificar el clasificador
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import json

print("=" * 80)
print("🔍 DIAGNÓSTICO DEL CLASIFICADOR")
print("=" * 80)
print()

# Rutas
base_path = Path(__file__).parent.parent
model_path = base_path / 'Entrega2' / 'models' / 'best_model.pkl'
scaler_path = base_path / 'Entrega2' / 'models' / 'scaler.pkl'
label_encoder_path = base_path / 'Entrega2' / 'models' / 'label_encoder.pkl'
preparation_info_path = base_path / 'Entrega2' / 'data' / 'preparation_info.json'
label_mapping_path = base_path / 'Entrega2' / 'data' / 'label_mapping.json'

print("📂 Verificando archivos...")
print(f"   Modelo: {model_path.exists()} - {model_path}")
print(f"   Scaler: {scaler_path.exists()} - {scaler_path}")
print(f"   Label encoder: {label_encoder_path.exists()} - {label_encoder_path}")
print(f"   Preparation info: {preparation_info_path.exists()} - {preparation_info_path}")
print(f"   Label mapping: {label_mapping_path.exists()} - {label_mapping_path}")
print()

# Cargar modelo
print("🤖 Cargando modelo...")
model = joblib.load(model_path)
print(f"   Tipo: {type(model).__name__}")
print(f"   Clases esperadas: {model.n_classes_ if hasattr(model, 'n_classes_') else 'N/A'}")
print()

# Cargar label encoder
print("🏷️  Cargando label encoder...")
label_encoder = joblib.load(label_encoder_path)
print(f"   Clases en label_encoder: {label_encoder.classes_}")
print(f"   Número de clases: {len(label_encoder.classes_)}")
print()

# Cargar label mapping
print("📋 Cargando label mapping...")
with open(label_mapping_path, 'r', encoding='utf-8') as f:
    label_mapping = json.load(f)

unique_activities = sorted(set(label_mapping.values()))
print(f"   Actividades únicas en mapping: {unique_activities}")
print(f"   Número de actividades: {len(unique_activities)}")
print()

# Mapeo de clases
print("🔗 Mapeo de clases (label_encoder → nombre actividad):")
activity_dict = {}
for i, activity in enumerate(unique_activities):
    if i < len(label_encoder.classes_):
        label = label_encoder.classes_[i]
        activity_dict[label] = activity
        print(f"   {label} → {activity}")

print()

# Cargar preparation info
print("📊 Información de preparación...")
with open(preparation_info_path, 'r') as f:
    prep_info = json.load(f)

print(f"   Número de features esperadas: {prep_info['n_features']}")
print(f"   Primeras 10 features: {prep_info['feature_names'][:10]}")
print()

# Cargar scaler
print("⚖️  Cargando scaler...")
scaler = joblib.load(scaler_path)
print(f"   Tipo: {type(scaler).__name__}")
print(f"   Features escaladas: {scaler.n_features_in_}")
print(f"   Mean (primeras 5): {scaler.mean_[:5]}")
print(f"   Scale (primeras 5): {scaler.scale_[:5]}")
print()

# Prueba con features dummy
print("🧪 Probando predicción con features dummy...")
print()

# Crear features dummy (todas en 0.5)
dummy_features = {}
for fname in prep_info['feature_names']:
    dummy_features[fname] = 0.5

# Preparar features en orden correcto
feature_values = [dummy_features[fname] for fname in prep_info['feature_names']]
X = np.array(feature_values).reshape(1, -1)

print(f"   Shape de features: {X.shape}")
print(f"   Primeros 5 valores: {X[0, :5]}")
print()

# Escalar
X_scaled = scaler.transform(X)
print(f"   Después de escalar (primeros 5): {X_scaled[0, :5]}")
print()

# Predecir
prediction = model.predict(X_scaled)[0]
probabilities = model.predict_proba(X_scaled)[0]

print("📊 Resultados de predicción:")
print(f"   Predicción (índice): {prediction}")
print(f"   Predicción (label): {label_encoder.inverse_transform([prediction])[0]}")
print(f"   Predicción (nombre): {activity_dict.get(label_encoder.inverse_transform([prediction])[0], 'DESCONOCIDO')}")
print()
print("   Probabilidades por clase:")
for i, (cls, prob) in enumerate(zip(label_encoder.classes_, probabilities)):
    activity_name = activity_dict.get(cls, str(cls))
    print(f"      {i}. {activity_name}: {prob:.2%}")
print()

# Verificar que todas las clases están cubiertas
print("=" * 80)
print("✅ VERIFICACIÓN FINAL")
print("=" * 80)
print()

if len(label_encoder.classes_) == len(unique_activities):
    print(f"✅ Número de clases coincide: {len(label_encoder.classes_)}")
else:
    print(f"⚠️  ADVERTENCIA: Desajuste de clases!")
    print(f"   Label encoder: {len(label_encoder.classes_)} clases")
    print(f"   Label mapping: {len(unique_activities)} actividades")

print()
print("🎯 Actividades que debería detectar el sistema:")
for i, activity in enumerate(unique_activities, 1):
    print(f"   {i}. {activity}")
print()
