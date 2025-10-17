import pandas as pd; import numpy as np; import os
from scipy.signal import savgol_filter

INPUT_PATH = "../data/01_landmarks"; OUTPUT_PATH = "../data/02_preprocessed"
if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

for file in os.listdir(INPUT_PATH):
    if not file.endswith('.csv'): continue
    
    df = pd.read_csv(os.path.join(INPUT_PATH, file))
    print(f"   Preprocesando: '{file}'")

    # 1. Suavizado de ruido (Jitter Removal)
    for col in df.columns:
        if col.startswith('x_') or col.startswith('y_') or col.startswith('z_'):
            # El filtro Savitzky-Golay suaviza la trayectoria de cada landmark
            df[col] = savgol_filter(df[col], window_length=5, polyorder=2)
            
    # 2. Normalización de Coordenadas
    # Hacemos que el esqueleto sea invariante a la posición y escala
    # Calculamos el centro del torso como punto de referencia
    df['center_x'] = (df['x_11'] + df['x_12']) / 2
    df['center_y'] = (df['y_11'] + df['y_12']) / 2
    
    # Restamos el centro a todas las coordenadas
    for i in range(33):
        df[f'x_{i}'] = df[f'x_{i}'] - df['center_x']
        df[f'y_{i}'] = df[f'y_{i}'] - df['center_y']
        
    df.drop(columns=['center_x', 'center_y'], inplace=True)
    
    df.to_csv(os.path.join(OUTPUT_PATH, file), index=False)

print("\n Preprocesamiento completado.")