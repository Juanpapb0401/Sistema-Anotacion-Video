"""
Verificar si el problema es el MODELO o la EXTRACCIÓN de features en tiempo real
"""
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from collections import Counter

base_path = Path(__file__).parent.parent

print("=" * 80)
print("🧪 VERIFICACIÓN: ¿Dónde está el problema?")
print("=" * 80)

# Cargar todo
print("\n1️⃣ Cargando componentes...")
model = joblib.load(base_path / "Entrega2" / "models" / "best_model.pkl")
scaler = joblib.load(base_path / "Entrega2" / "models" / "scaler.pkl")
label_encoder = joblib.load(base_path / "Entrega2" / "models" / "label_encoder.pkl")

print(f"   Classes: {list(label_encoder.classes_)}")

# Test con test.csv (datos BUENOS conocidos)
print("\n2️⃣ Test con DATOS DE ENTRENAMIENTO (test.csv)...")
test_df = pd.read_csv(base_path / "Entrega2" / "data" / "test.csv")
X_test = test_df.drop(['label', 'person', 'video_id', 'video_speed'], axis=1)
y_test = test_df['label'].values

# Tomar muestra de cada clase
print("\n   Tomando 20 muestras de cada clase...")
samples_per_class = {}

if isinstance(y_test[0], (int, np.integer)):
    y_test_names = label_encoder.inverse_transform(y_test)
else:
    y_test_names = y_test

for cls in sorted(np.unique(y_test_names)):
    idx = np.where(y_test_names == cls)[0][:20]  # Primeras 20
    samples_per_class[cls] = {
        'indices': idx,
        'X': X_test.iloc[idx],
        'y': y_test_names[idx]
    }
    print(f"      {cls}: {len(idx)} muestras")

# Predecir clase por clase
print("\n3️⃣ PREDICIENDO POR CLASE:")
print("=" * 80)

all_correct = True

for cls, data in samples_per_class.items():
    X_sample = scaler.transform(data['X'])
    preds = model.predict(X_sample)
    pred_names = label_encoder.inverse_transform(preds)
    
    # Contar predicciones
    counts = Counter(pred_names)
    correct = sum(1 for p in pred_names if p == cls)
    accuracy = (correct / len(pred_names)) * 100
    
    print(f"\n🏷️  Clase: {cls}")
    print(f"   Accuracy: {accuracy:.1f}% ({correct}/{len(pred_names)})")
    print(f"   Predicciones:")
    
    for pred, count in counts.most_common():
        pct = (count / len(pred_names)) * 100
        marker = "✅" if pred == cls else "❌"
        print(f"      {marker} {pred:25s}: {count:2d} ({pct:5.1f}%)")
    
    if accuracy < 50:
        all_correct = False
        print(f"   ⚠️  PROBLEMA: El modelo NO reconoce bien '{cls}' ni en test.csv")

# Conclusión
print("\n" + "=" * 80)
print("🔍 CONCLUSIÓN:")
print("=" * 80)

if all_correct:
    print("\n✅ El MODELO funciona correctamente con datos de test.csv")
    print("\n❌ El problema está en la EXTRACCIÓN DE FEATURES en tiempo real:")
    print("   → Las features de la cámara NO coinciden con las de entrenamiento")
    print("   → video_processor.py genera features diferentes")
    print("   → Scaler no está aplicado correctamente")
    print("\n💡 SOLUCIÓN:")
    print("   1. Ejecutar diagnose_features.py para ver qué features son diferentes")
    print("   2. Verificar que video_processor genera las 147 features correctas")
    print("   3. Verificar nombres de features (x_0, y_0, etc.)")
else:
    print("\n❌ El MODELO ya no funciona ni con datos de test.csv")
    print("\n   Posibles causas:")
    print("   → Modelo guardado incorrectamente")
    print("   → Scaler guardado incorrectamente")  
    print("   → Label encoder guardado incorrectamente")
    print("   → Archivos corruptos")
    print("\n💡 SOLUCIÓN:")
    print("   → RE-ENTRENAR el modelo desde cero")
    print("   → Ejecutar 04_model_training.py nuevamente")

print("\n" + "=" * 80)
