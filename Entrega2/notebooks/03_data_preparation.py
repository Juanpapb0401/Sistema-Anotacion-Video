"""
Script para preparar los datos para entrenamiento de modelos
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Rutas
DATA_PATH = "../data"
OUTPUT_PATH = "../data"
MODELS_PATH = "../models"


def load_data():
    """Carga el dataset principal"""
    data_path = Path(DATA_PATH)
    df = pd.read_csv(data_path / "labeled_dataset_main.csv")
    
    print(f"📂 Dataset cargado: {len(df):,} frames")
    print(f"   Clases: {df['label'].unique().tolist()}")
    print(f"   Personas: {df['person'].unique().tolist()}")
    
    return df


def create_temporal_features(df):
    """
    Crea features temporales (velocidades y aceleraciones)
    """
    print("\n🔄 Creando features temporales...")
    
    # Crear copia
    df = df.copy()
    df = df.sort_values(['person', 'video_id', 'frame'])
    
    # Features base
    angle_features = [col for col in df.columns if 'angle' in col]
    distance_features = [col for col in df.columns if 'dist' in col]
    
    # Calcular velocidades (diferencia entre frames consecutivos)
    for feature in angle_features + distance_features:
        df[f'{feature}_velocity'] = df.groupby(['person', 'video_id'])[feature].diff()
    
    # Calcular aceleraciones (diferencia de velocidades)
    velocity_features = [col for col in df.columns if 'velocity' in col]
    for feature in velocity_features:
        df[f'{feature}_accel'] = df.groupby(['person', 'video_id'])[feature].diff()
    
    # Rellenar NaN en las primeras filas de cada video con 0
    df = df.fillna(0)
    
    print(f"✅ Features temporales creadas")
    print(f"   Total de features: {len([col for col in df.columns if col not in ['frame', 'label_raw', 'label', 'person', 'video_id', 'video_speed']])}")
    
    return df


def prepare_features_and_labels(df):
    """
    Separa features y labels, codifica labels
    """
    print("\n📊 Preparando features y labels...")
    
    # Eliminar frames sin etiqueta
    df = df.dropna(subset=['label'])
    
    # Columnas a excluir
    exclude_cols = ['frame', 'label_raw', 'label', 'person', 'video_id', 'video_speed']
    
    # Features (todas las columnas numéricas)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    
    # Labels
    y = df['label']
    
    # Codificar labels a números
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Metadata (persona, video_id)
    metadata = df[['person', 'video_id', 'video_speed']]
    
    print(f"✅ Features: {X.shape[1]} columnas")
    print(f"   Labels: {len(label_encoder.classes_)} clases")
    print(f"   Muestras: {len(X):,}")
    
    # Mostrar distribución de clases
    print("\n   Distribución de clases:")
    for i, label in enumerate(label_encoder.classes_):
        count = (y_encoded == i).sum()
        pct = count / len(y_encoded) * 100
        print(f"   - {label}: {count:,} ({pct:.1f}%)")
    
    return X, y_encoded, label_encoder, feature_cols, metadata


def split_data(X, y, metadata, test_size=0.15, val_size=0.15, random_state=42):
    """
    Divide los datos en train, validation y test de forma estratificada
    """
    print(f"\n✂️  Dividiendo datos (train/val/test)...")
    
    # Calcular tamaño de validación relativo al conjunto de entrenamiento
    val_size_adjusted = val_size / (1 - test_size)
    
    # Primero separar test
    X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
        X, y, metadata,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Luego separar train y validation
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_temp, y_temp, meta_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=random_state
    )
    
    print(f"✅ División completada:")
    print(f"   Train: {len(X_train):,} muestras ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validation: {len(X_val):,} muestras ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test):,} muestras ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test


def normalize_features(X_train, X_val, X_test):
    """
    Normaliza las features usando StandardScaler
    """
    print("\n📏 Normalizando features...")
    
    scaler = StandardScaler()
    
    # Ajustar solo con datos de entrenamiento
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Normalización completada")
    print(f"   Media: {X_train_scaled.mean():.4f}")
    print(f"   Std: {X_train_scaled.std():.4f}")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def balance_classes(X_train, y_train, random_state=42):
    """
    Balancea las clases usando SMOTE
    """
    print("\n⚖️  Balanceando clases con SMOTE...")
    
    # Contar muestras antes
    unique, counts = np.unique(y_train, return_counts=True)
    print("   Distribución antes de SMOTE:")
    for label, count in zip(unique, counts):
        print(f"   - Clase {label}: {count:,}")
    
    # Aplicar SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Contar muestras después
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    print("\n   Distribución después de SMOTE:")
    for label, count in zip(unique, counts):
        print(f"   - Clase {label}: {count:,}")
    
    print(f"\n✅ Balanceo completado")
    print(f"   Muestras antes: {len(X_train):,}")
    print(f"   Muestras después: {len(X_train_balanced):,}")
    print(f"   Incremento: {(len(X_train_balanced) - len(X_train)) / len(X_train) * 100:.1f}%")
    
    return X_train_balanced, y_train_balanced


def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test,
                       meta_train, meta_val, meta_test,
                       feature_cols, label_encoder, scaler):
    """
    Guarda los datos procesados
    """
    print("\n💾 Guardando datos procesados...")
    
    output_path = Path(OUTPUT_PATH)
    models_path = Path(MODELS_PATH)
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Convertir arrays a dataframes con nombres de columnas
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df['label'] = y_train
    train_df = pd.concat([train_df, meta_train.reset_index(drop=True)], axis=1)
    
    val_df = pd.DataFrame(X_val, columns=feature_cols)
    val_df['label'] = y_val
    val_df = pd.concat([val_df, meta_val.reset_index(drop=True)], axis=1)
    
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df['label'] = y_test
    test_df = pd.concat([test_df, meta_test.reset_index(drop=True)], axis=1)
    
    # Guardar CSVs
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "validation.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    
    print(f"✅ Datasets guardados:")
    print(f"   - train.csv")
    print(f"   - validation.csv")
    print(f"   - test.csv")
    
    # Guardar objetos de preprocesamiento
    joblib.dump(label_encoder, models_path / "label_encoder.pkl")
    joblib.dump(scaler, models_path / "scaler.pkl")
    
    print(f"\n✅ Objetos de preprocesamiento guardados:")
    print(f"   - label_encoder.pkl")
    print(f"   - scaler.pkl")
    
    # Guardar información de preparación
    prep_info = {
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'n_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_.tolist(),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'total_samples': len(train_df) + len(val_df) + len(test_df)
    }
    
    with open(output_path / "preparation_info.json", 'w') as f:
        json.dump(prep_info, f, indent=2)
    
    print(f"✅ Información de preparación guardada")


def generate_preparation_report(df_original, X_train, X_val, X_test, 
                                y_train, y_val, y_test, label_encoder):
    """
    Genera un reporte de texto con información de la preparación
    """
    print("\n📝 Generando reporte de preparación...")
    
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("REPORTE DE PREPARACIÓN DE DATOS")
    report_lines.append("Sistema de Anotación de Video - Entrega 2")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Información general
    report_lines.append("📊 INFORMACIÓN GENERAL")
    report_lines.append("-" * 80)
    report_lines.append(f"Dataset original: {len(df_original):,} frames")
    report_lines.append(f"Features extraídas: {X_train.shape[1]}")
    report_lines.append(f"Clases: {len(label_encoder.classes_)}")
    report_lines.append(f"Clases: {', '.join(label_encoder.classes_)}")
    report_lines.append("")
    
    # División de datos
    report_lines.append("✂️  DIVISIÓN DE DATOS")
    report_lines.append("-" * 80)
    total = len(X_train) + len(X_val) + len(X_test)
    report_lines.append(f"Train: {len(X_train):,} ({len(X_train)/total*100:.1f}%)")
    report_lines.append(f"Validation: {len(X_val):,} ({len(X_val)/total*100:.1f}%)")
    report_lines.append(f"Test: {len(X_test):,} ({len(X_test)/total*100:.1f}%)")
    report_lines.append(f"Total: {total:,}")
    report_lines.append("")
    
    # Distribución por conjunto
    for name, y_set in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
        report_lines.append(f"\n{name} - Distribución de clases:")
        unique, counts = np.unique(y_set, return_counts=True)
        for label_idx, count in zip(unique, counts):
            label_name = label_encoder.classes_[label_idx]
            pct = count / len(y_set) * 100
            report_lines.append(f"  {label_name}: {count:,} ({pct:.1f}%)")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Guardar reporte
    report_text = "\n".join(report_lines)
    report_file = Path(OUTPUT_PATH) / "preparation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✅ Reporte guardado: preparation_report.txt")
    
    return report_text


def main():
    """Función principal"""
    print("🚀 Iniciando Preparación de Datos\n")
    print("=" * 80)
    
    # 1. Cargar datos
    df = load_data()
    
    # 2. Crear features temporales
    df = create_temporal_features(df)
    
    # 3. Preparar features y labels
    X, y, label_encoder, feature_cols, metadata = prepare_features_and_labels(df)
    
    # 4. Dividir datos
    X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = split_data(
        X, y, metadata
    )
    
    # 5. Normalizar features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_features(
        X_train, X_val, X_test
    )
    
    # 6. Balancear clases (solo en entrenamiento)
    X_train_balanced, y_train_balanced = balance_classes(X_train_scaled, y_train)
    
    # 7. Guardar datos procesados (con train balanceado)
    save_processed_data(
        X_train_balanced, X_val_scaled, X_test_scaled,
        y_train_balanced, y_val, y_test,
        meta_train, meta_val, meta_test,
        feature_cols, label_encoder, scaler
    )
    
    # 8. Generar reporte
    report = generate_preparation_report(
        df, X_train_balanced, X_val_scaled, X_test_scaled,
        y_train_balanced, y_val, y_test, label_encoder
    )
    
    print("\n" + "=" * 80)
    print(report)
    
    print("\n✅ Preparación de datos completada!")
    print(f"📁 Archivos generados en: {OUTPUT_PATH}/")
    print("   - train.csv")
    print("   - validation.csv")
    print("   - test.csv")
    print("   - preparation_info.json")
    print("   - preparation_report.txt")
    print(f"\n📁 Objetos guardados en: {MODELS_PATH}/")
    print("   - label_encoder.pkl")
    print("   - scaler.pkl")


if __name__ == "__main__":
    main()
