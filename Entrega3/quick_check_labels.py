"""
Quick check: ¿train.csv tiene strings o números?
"""
import pandas as pd
from pathlib import Path

train_path = Path(__file__).parent.parent / "Entrega2" / "data" / "train.csv"
df = pd.read_csv(train_path)

print("=" * 60)
print("🔍 CONTENIDO DE train.csv - COLUMNA 'label'")
print("=" * 60)
print(f"\nTipo de dato: {df['label'].dtype}")
print(f"Valores únicos: {sorted(df['label'].unique())}")
print(f"\nPrimeros 10 valores:")
print(df['label'].head(10).tolist())
print(f"\nÚltimos 10 valores:")
print(df['label'].tail(10).tolist())

if df['label'].dtype == 'int64' or df['label'].dtype == 'float64':
    print("\n❌ PROBLEMA: Labels son NÚMEROS, deberían ser STRINGS")
    print("   El label_encoder.inverse_transform() no funcionará correctamente")
else:
    print("\n✅ Labels son STRINGS (correcto)")
