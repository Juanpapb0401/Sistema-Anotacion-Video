"""
Test SIMPLE: ¿El modelo predice correctamente con datos de entrenamiento?
"""
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

base_path = Path(__file__).parent.parent

print("=" * 80)
print("🧪 TEST: ¿El modelo predice correctamente?")
print("=" * 80)

# Cargar modelo, scaler y label_encoder
print("\n1️⃣ Cargando modelo...")
model = joblib.load(base_path / "Entrega2" / "models" / "best_model.pkl")
scaler = joblib.load(base_path / "Entrega2" / "models" / "scaler.pkl")
label_encoder = joblib.load(base_path / "Entrega2" / "models" / "label_encoder.pkl")

print(f"   ✅ Modelo cargado")
print(f"   Label encoder classes: {label_encoder.classes_}")

# Cargar test.csv
print("\n2️⃣ Cargando test.csv...")
test_df = pd.read_csv(base_path / "Entrega2" / "data" / "test.csv")
X_test = test_df.drop(['label', 'person', 'video_id', 'video_speed'], axis=1)
y_test = test_df['label'].values

print(f"   ✅ {len(X_test)} muestras cargadas")
print(f"   Labels tipo: {type(y_test[0])}, ejemplos: {y_test[:5]}")

# Escalar features
print("\n3️⃣ Escalando features...")
X_test_scaled = scaler.transform(X_test)

# Predecir en primeras 100 muestras
print("\n4️⃣ Prediciendo primeras 100 muestras...")
n_samples = min(100, len(X_test))
X_sample = X_test_scaled[:n_samples]
y_sample = y_test[:n_samples]

predictions = model.predict(X_sample)
probabilities = model.predict_proba(X_sample)

print(f"   ✅ Predicciones realizadas")
print(f"   Predictions tipo: {type(predictions[0])}, ejemplos: {predictions[:5]}")

# ¿Los labels en test.csv son números o strings?
print("\n5️⃣ Análisis de labels:")
if isinstance(y_sample[0], str):
    print("   ✅ Labels en CSV son STRINGS")
    print(f"   Ejemplos: {y_sample[:5]}")
    
    # Convertir predicciones numéricas a strings
    pred_names = label_encoder.inverse_transform(predictions)
    print(f"\n   Predicciones como strings:")
    print(f"   Ejemplos: {pred_names[:5]}")
    
    # Comparar
    correct = sum(pred_names == y_sample)
    print(f"\n   ✅ Accuracy en muestra: {correct}/{n_samples} ({100*correct/n_samples:.1f}%)")
    
else:
    print("   ⚠️  Labels en CSV son NÚMEROS")
    print(f"   Ejemplos: {y_sample[:5]}")
    
    # Labels ya son números
    correct = sum(predictions == y_sample)
    print(f"\n   ✅ Accuracy en muestra: {correct}/{n_samples} ({100*correct/n_samples:.1f}%)")
    
    # Convertir a nombres
    true_names = label_encoder.inverse_transform(y_sample)
    pred_names = label_encoder.inverse_transform(predictions)
    
    print(f"\n   Labels verdaderos como strings:")
    print(f"   Ejemplos: {true_names[:5]}")
    print(f"\n   Predicciones como strings:")
    print(f"   Ejemplos: {pred_names[:5]}")

# Mostrar algunos ejemplos detallados
print("\n" + "=" * 80)
print("📋 EJEMPLOS DETALLADOS (primeros 10):")
print("=" * 80)

for i in range(min(10, n_samples)):
    if isinstance(y_sample[0], str):
        true_label = y_sample[i]
        pred_label = label_encoder.inverse_transform([predictions[i]])[0]
    else:
        true_label = label_encoder.inverse_transform([y_sample[i]])[0]
        pred_label = label_encoder.inverse_transform([predictions[i]])[0]
    
    pred_probs = probabilities[i]
    max_prob = pred_probs[predictions[i]]
    
    status = "✅" if true_label == pred_label else "❌"
    
    print(f"{i+1:2d}. {status} Real: {true_label:25s} | Pred: {pred_label:25s} (conf: {max_prob:.3f})")

print("\n" + "=" * 80)
