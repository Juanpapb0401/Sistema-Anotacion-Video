"""
Verificar el orden EXACTO de clases en label_encoder y train.csv
"""
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

base_path = Path(__file__).parent.parent

print("=" * 80)
print("🔍 VERIFICACIÓN DE ORDEN DE CLASES")
print("=" * 80)

# 1. Cargar label_encoder.pkl
print("\n1️⃣ Label Encoder guardado:")
label_encoder_path = base_path / "Entrega2" / "models" / "label_encoder.pkl"
label_encoder = joblib.load(label_encoder_path)

print(f"   Tipo: {type(label_encoder)}")
print(f"   Classes attribute:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"      {i} → {cls}")

# 2. Cargar train.csv y ver las clases
print("\n2️⃣ Clases en train.csv:")
train_path = base_path / "Entrega2" / "data" / "train.csv"
df = pd.read_csv(train_path)

unique_labels = sorted(df['label'].unique())
print(f"   Labels únicos (ordenados alfabéticamente):")
for i, label in enumerate(unique_labels):
    count = (df['label'] == label).sum()
    print(f"      {i} → {label} ({count} muestras)")

# 3. Simular lo que hace LabelEncoder.fit_transform
print("\n3️⃣ Simulación de fit_transform:")
test_encoder = LabelEncoder()
test_labels = df['label'].values
test_encoded = test_encoder.fit_transform(test_labels)

print(f"   Clases después de fit_transform:")
for i, cls in enumerate(test_encoder.classes_):
    count = (test_encoded == i).sum()
    print(f"      {i} → {cls} ({count} muestras)")

# 4. Verificar mapeo inverso
print("\n4️⃣ Verificación de inverse_transform:")
test_numbers = [0, 1, 2, 3, 4]
test_names = label_encoder.inverse_transform(test_numbers)
print(f"   Mapeo número → nombre:")
for num, name in zip(test_numbers, test_names):
    print(f"      {num} → {name}")

# 5. Verificar si train.csv tiene encodings numéricos
print("\n5️⃣ Verificación de valores en train.csv:")
print(f"   Tipo de datos de 'label': {df['label'].dtype}")
print(f"   Muestra de labels:")
print(f"      Primeros 5: {df['label'].head().tolist()}")
print(f"      Últimos 5: {df['label'].tail().tolist()}")

# 6. ¿Los labels en train.csv son strings o números?
if df['label'].dtype == 'object':
    print("\n   ✅ Labels son STRINGS (nombres de actividades)")
else:
    print("\n   ⚠️  Labels son NÚMEROS (ya encodificados)")
    print(f"   Rango de valores: {df['label'].min()} a {df['label'].max()}")

print("\n" + "=" * 80)
