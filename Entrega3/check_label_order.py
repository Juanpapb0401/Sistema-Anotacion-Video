"""
Script rápido para verificar el orden de clases en el label_encoder
"""
import joblib
from pathlib import Path

label_encoder_path = Path("../Entrega2/models/label_encoder.pkl")
label_encoder = joblib.load(label_encoder_path)

print("=" * 60)
print("📋 ORDEN DE CLASES EN EL LABEL_ENCODER")
print("=" * 60)
print()

for i, cls in enumerate(label_encoder.classes_):
    print(f"   {i} → {cls}")

print()
