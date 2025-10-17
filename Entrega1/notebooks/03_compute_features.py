import pandas as pd; import numpy as np; import os

INPUT_PATH = "../data/02_preprocessed"; OUTPUT_PATH = "../data/03_features"
if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return angle if angle <= 180 else 360-angle

for file in os.listdir(INPUT_PATH):
    if not file.endswith('.csv'): continue
    df = pd.read_csv(os.path.join(INPUT_PATH, file))
    print(f"   Calculando features para: '{file}'")
    
    feature_df = pd.DataFrame()
    feature_df['frame'] = df['frame']
    
    # Calculamos ángulos clave para cada fotograma
    feature_df['left_knee_angle'] = df.apply(lambda r: calculate_angle([r.x_23,r.y_23],[r.x_25,r.y_25],[r.x_27,r.y_27]), axis=1)
    feature_df['right_knee_angle'] = df.apply(lambda r: calculate_angle([r.x_24,r.y_24],[r.x_26,r.y_26],[r.x_28,r.y_28]), axis=1)
    feature_df['left_elbow_angle'] = df.apply(lambda r: calculate_angle([r.x_11,r.y_11],[r.x_13,r.y_13],[r.x_15,r.y_15]), axis=1)
    feature_df['right_elbow_angle'] = df.apply(lambda r: calculate_angle([r.x_12,r.y_12],[r.x_14,r.y_14],[r.x_16,r.y_16]), axis=1)
    
    # Una métrica simple de movimiento: distancia de la muñeca al hombro
    feature_df['left_wrist_dist'] = np.linalg.norm(df[['x_15','y_15']].values - df[['x_11','y_11']].values, axis=1)

    feature_df.to_csv(os.path.join(OUTPUT_PATH, file), index=False)

print("\n Cómputo de características completado.")