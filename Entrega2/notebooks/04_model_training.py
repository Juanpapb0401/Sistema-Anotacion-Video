"""
Script para entrenar múltiples modelos de clasificación con ajuste de hiperparámetros
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import time
from datetime import datetime

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

# Intentar importar XGBoost (opcional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("️  XGBoost no disponible. Solo se entrenarán SVM y Random Forest.")

# Rutas
DATA_PATH = "../data"
MODELS_PATH = "../models"


def load_prepared_data():
    """Carga los datos preparados"""
    print(" Cargando datos preparados para poder entrenar...")
    
    data_path = Path(DATA_PATH)
    
    # Cargar datasets
    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "validation.csv")
    
    # Separar features y labels
    feature_cols = [col for col in train_df.columns if col not in ['label', 'person', 'video_id', 'video_speed']]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    
    # Cargar objetos de preprocesamiento
    models_path = Path(MODELS_PATH)
    label_encoder = joblib.load(models_path / "label_encoder.pkl")
    
    print(f" Datos cargados:")
    print(f"   Train: {len(X_train):,} muestras, {X_train.shape[1]} features")
    print(f"   Validation: {len(X_val):,} muestras")
    print(f"   Clases: {label_encoder.classes_.tolist()}")
    
    return X_train, y_train, X_val, y_val, label_encoder, feature_cols


def train_svm(X_train, y_train, X_val, y_val):
    """
    Entrena un modelo SVM con ajuste de hiperparámetros
    """
    print("\n" + "="*80)
    print(" Entrenando SVM (Support Vector Machine)")
    print("="*80)
    
    # Definir espacio de búsqueda de hiperparámetros
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    
    print(" Búsqueda de hiperparámetros con GridSearchCV...")
    print(f"   Espacio de búsqueda: {len(param_grid['C']) * len(param_grid['kernel']) * len(param_grid['gamma'])} combinaciones")
    
    start_time = time.time()
    
    # Grid Search con validación cruzada
    svm = SVC(random_state=42, probability=True)
    grid_search = GridSearchCV(
        svm, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Mejor modelo
    best_model = grid_search.best_estimator_
    
    print(f"\n Entrenamiento completado en {training_time:.2f}s")
    print(f"   Mejores parámetros: {grid_search.best_params_}")
    print(f"   Mejor score CV: {grid_search.best_score_:.4f}")
    
    # Evaluar en validación
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"\n Resultados en Validación:")
    print(f"   Accuracy: {val_accuracy:.4f}")
    print(f"   F1-Score: {val_f1:.4f}")
    
    results = {
        'model_name': 'SVM',
        'best_params': grid_search.best_params_,
        'best_cv_score': float(grid_search.best_score_),
        'val_accuracy': float(val_accuracy),
        'val_f1_score': float(val_f1),
        'training_time': float(training_time)
    }
    
    return best_model, results


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Entrena un modelo Random Forest con ajuste de hiperparámetros
    """
    print("\n" + "="*80)
    print("🌲 Entrenando Random Forest")
    print("="*80)
    
    # Definir espacio de búsqueda de hiperparámetros
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print(" Búsqueda de hiperparámetros con GridSearchCV...")
    print(f"   Espacio de búsqueda: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])} combinaciones")
    
    start_time = time.time()
    
    # Grid Search con validación cruzada
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Mejor modelo
    best_model = grid_search.best_estimator_
    
    print(f"\n Entrenamiento completado en {training_time:.2f}s")
    print(f"   Mejores parámetros: {grid_search.best_params_}")
    print(f"   Mejor score CV: {grid_search.best_score_:.4f}")
    
    # Evaluar en validación
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"\n Resultados en Validación:")
    print(f"   Accuracy: {val_accuracy:.4f}")
    print(f"   F1-Score: {val_f1:.4f}")
    
    results = {
        'model_name': 'Random Forest',
        'best_params': grid_search.best_params_,
        'best_cv_score': float(grid_search.best_score_),
        'val_accuracy': float(val_accuracy),
        'val_f1_score': float(val_f1),
        'training_time': float(training_time)
    }
    
    return best_model, results


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Entrena un modelo XGBoost con ajuste de hiperparámetros
    """
    print("\n" + "="*80)
    print(" Entrenando XGBoost")
    print("="*80)
    
    # Definir espacio de búsqueda de hiperparámetros
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    print(" Búsqueda de hiperparámetros con GridSearchCV...")
    print(f"   Espacio de búsqueda: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree'])} combinaciones")
    
    start_time = time.time()
    
    # Grid Search con validación cruzada
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    grid_search = GridSearchCV(
        xgb_model, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Mejor modelo
    best_model = grid_search.best_estimator_
    
    print(f"\n Entrenamiento completado en {training_time:.2f}s")
    print(f"   Mejores parámetros: {grid_search.best_params_}")
    print(f"   Mejor score CV: {grid_search.best_score_:.4f}")
    
    # Evaluar en validación
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    print(f"\n Resultados en Validación:")
    print(f"   Accuracy: {val_accuracy:.4f}")
    print(f"   F1-Score: {val_f1:.4f}")
    
    results = {
        'model_name': 'XGBoost',
        'best_params': grid_search.best_params_,
        'best_cv_score': float(grid_search.best_score_),
        'val_accuracy': float(val_accuracy),
        'val_f1_score': float(val_f1),
        'training_time': float(training_time)
    }
    
    return best_model, results


def save_models_and_results(models_dict, results_list):
    """
    Guarda los modelos entrenados y los resultados
    """
    print("\n Guardando modelos y resultados...")
    
    models_path = Path(MODELS_PATH)
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Guardar cada modelo
    for model_name, model in models_dict.items():
        model_file = models_path / f"{model_name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, model_file)
        print(f"    {model_name} guardado: {model_file.name}")
    
    # Identificar el mejor modelo
    best_model_info = max(results_list, key=lambda x: x['val_f1_score'])
    best_model_name = best_model_info['model_name']
    best_model = models_dict[best_model_name]
    
    # Guardar el mejor modelo por separado
    joblib.dump(best_model, models_path / "best_model.pkl")
    print(f"\n    Mejor modelo: {best_model_name} (F1={best_model_info['val_f1_score']:.4f})")
    print(f"    Guardado como: best_model.pkl")
    
    # Guardar resultados en JSON
    training_results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': results_list,
        'best_model': {
            'name': best_model_name,
            'f1_score': best_model_info['val_f1_score'],
            'accuracy': best_model_info['val_accuracy']
        }
    }
    
    results_file = models_path / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print(f"    Resultados guardados: training_results.json")
    
    return best_model_name, best_model_info


def generate_training_report(results_list, best_model_name, best_model_info):
    """
    Genera un reporte de texto con los resultados del entrenamiento
    """
    print("\n Generando reporte de entrenamiento...")
    
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("REPORTE DE ENTRENAMIENTO DE MODELOS")
    report_lines.append("Sistema de Anotación de Video - Entrega 2")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Resumen de todos los modelos
    report_lines.append(" RESUMEN DE MODELOS")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Modelo':<20} {'Accuracy':<12} {'F1-Score':<12} {'Tiempo (s)':<12}")
    report_lines.append("-" * 80)
    
    for result in sorted(results_list, key=lambda x: x['val_f1_score'], reverse=True):
        model_name = result['model_name']
        accuracy = result['val_accuracy']
        f1 = result['val_f1_score']
        time_taken = result['training_time']
        
        marker = "" if model_name == best_model_name else "  "
        report_lines.append(f"{marker} {model_name:<18} {accuracy:<12.4f} {f1:<12.4f} {time_taken:<12.2f}")
    
    report_lines.append("")
    
    # Detalles de cada modelo
    report_lines.append(" DETALLES DE MODELOS")
    report_lines.append("-" * 80)
    
    for result in results_list:
        report_lines.append(f"\n{result['model_name']}:")
        report_lines.append(f"  Mejores hiperparámetros:")
        for param, value in result['best_params'].items():
            report_lines.append(f"    - {param}: {value}")
        report_lines.append(f"  Score CV (5-fold): {result['best_cv_score']:.4f}")
        report_lines.append(f"  Accuracy (validación): {result['val_accuracy']:.4f}")
        report_lines.append(f"  F1-Score (validación): {result['val_f1_score']:.4f}")
        report_lines.append(f"  Tiempo de entrenamiento: {result['training_time']:.2f}s")
    
    # Mejor modelo
    report_lines.append("")
    report_lines.append(" MEJOR MODELO")
    report_lines.append("-" * 80)
    report_lines.append(f"Modelo seleccionado: {best_model_name}")
    report_lines.append(f"F1-Score: {best_model_info['val_f1_score']:.4f}")
    report_lines.append(f"Accuracy: {best_model_info['val_accuracy']:.4f}")
    report_lines.append("")
    report_lines.append("El mejor modelo ha sido guardado como 'best_model.pkl'")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Guardar reporte
    report_text = "\n".join(report_lines)
    report_file = Path(MODELS_PATH) / "training_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f" Reporte guardado: training_report.txt")
    
    return report_text


def main():
    """Función principal"""
    print(" Iniciando Entrenamiento de Modelos\n")
    print("=" * 80)
    
    # 1. Cargar datos preparados
    X_train, y_train, X_val, y_val, label_encoder, feature_cols = load_prepared_data()
    
    # 2. Entrenar modelos
    models_dict = {}
    results_list = []
    
    # SVM
    svm_model, svm_results = train_svm(X_train, y_train, X_val, y_val)
    models_dict['SVM'] = svm_model
    results_list.append(svm_results)
    
    # Random Forest
    rf_model, rf_results = train_random_forest(X_train, y_train, X_val, y_val)
    models_dict['Random Forest'] = rf_model
    results_list.append(rf_results)
    
    # XGBoost (solo si está disponible)
    if XGBOOST_AVAILABLE:
        xgb_model, xgb_results = train_xgboost(X_train, y_train, X_val, y_val)
        models_dict['XGBoost'] = xgb_model
        results_list.append(xgb_results)
    else:
        print("\n️  Saltando entrenamiento de XGBoost (no disponible)")
    
    # 3. Guardar modelos y resultados
    best_model_name, best_model_info = save_models_and_results(models_dict, results_list)
    
    # 4. Generar reporte
    report = generate_training_report(results_list, best_model_name, best_model_info)
    
    print("\n" + "=" * 80)
    print(report)
    
    print("\n Entrenamiento de modelos completado!")
    print(f" Modelos guardados en: {MODELS_PATH}/")
    print("   - svm_model.pkl")
    print("   - random_forest_model.pkl")
    if XGBOOST_AVAILABLE:
        print("   - xgboost_model.pkl")
    print("   - best_model.pkl")
    print("   - training_results.json")
    print("   - training_report.txt")


if __name__ == "__main__":
    main()
