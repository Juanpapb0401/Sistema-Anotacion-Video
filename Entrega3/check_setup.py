"""
Script de verificación para asegurar que todo esté listo para la interfaz
"""

import sys
from pathlib import Path

def check_file(filepath, description):
    """Verifica si un archivo existe"""
    if filepath.exists():
        size = filepath.stat().st_size / (1024 * 1024)  # MB
        print(f"   ✅ {description}: {filepath.name} ({size:.2f} MB)")
        return True
    else:
        print(f"   ❌ {description}: {filepath.name} - NO ENCONTRADO")
        return False


def main():
    print("🔍 VERIFICACIÓN DE CONFIGURACIÓN PARA INTERFAZ GRÁFICA")
    print("=" * 80)
    
    base_path = Path(__file__).parent.parent
    all_ok = True
    
    # 1. Verificar modelos entrenados
    print("\n📦 1. MODELOS ENTRENADOS (Entrega2/models/)")
    print("-" * 80)
    
    models_path = base_path / "Entrega2" / "models"
    
    required_files = [
        (models_path / "best_model.pkl", "Mejor modelo"),
        (models_path / "random_forest_model.pkl", "Random Forest"),
        (models_path / "scaler.pkl", "Scaler (normalización)"),
        (models_path / "label_encoder.pkl", "Label Encoder"),
        (models_path / "training_results.json", "Resultados de entrenamiento"),
        (models_path / "training_report.txt", "Reporte de entrenamiento")
    ]
    
    for filepath, desc in required_files:
        if not check_file(filepath, desc):
            all_ok = False
    
    # Verificar modelos opcionales
    optional_files = [
        (models_path / "svm_model.pkl", "SVM (opcional)"),
        (models_path / "xgboost_model.pkl", "XGBoost (opcional)")
    ]
    
    print("\n   Modelos opcionales:")
    for filepath, desc in optional_files:
        check_file(filepath, desc)
    
    # 2. Verificar datos
    print("\n📊 2. DATOS PREPARADOS (Entrega2/data/)")
    print("-" * 80)
    
    data_path = base_path / "Entrega2" / "data"
    
    data_files = [
        (data_path / "train.csv", "Dataset de entrenamiento"),
        (data_path / "validation.csv", "Dataset de validación"),
        (data_path / "test.csv", "Dataset de prueba"),
        (data_path / "preparation_info.json", "Información de features"),
        (data_path / "label_mapping.json", "Mapeo de etiquetas")
    ]
    
    for filepath, desc in data_files:
        if not check_file(filepath, desc):
            all_ok = False
    
    # 3. Verificar scripts de interfaz
    print("\n🎨 3. SCRIPTS DE INTERFAZ (Entrega3/real_time/)")
    print("-" * 80)
    
    interface_path = base_path / "Entrega3" / "real_time"
    
    interface_files = [
        (interface_path / "real_time_app.py", "Aplicación Streamlit"),
        (interface_path / "video_processor.py", "Procesador de video"),
        (interface_path / "activity_classifier.py", "Clasificador")
    ]
    
    for filepath, desc in interface_files:
        if not check_file(filepath, desc):
            all_ok = False
    
    # 4. Verificar dependencias
    print("\n📚 4. DEPENDENCIAS DE PYTHON")
    print("-" * 80)
    
    dependencies = [
        ("streamlit", "Interfaz gráfica web"),
        ("cv2", "OpenCV - Procesamiento de video"),
        ("mediapipe", "Detección de pose"),
        ("plotly", "Gráficos interactivos"),
        ("numpy", "Computación numérica"),
        ("pandas", "Manejo de datos"),
        ("sklearn", "Scikit-learn - Machine Learning"),
        ("joblib", "Serialización de modelos")
    ]
    
    missing_deps = []
    for module_name, desc in dependencies:
        try:
            if module_name == "cv2":
                import cv2
            elif module_name == "sklearn":
                import sklearn
            else:
                __import__(module_name)
            print(f"   ✅ {desc}: {module_name}")
        except ImportError:
            print(f"   ❌ {desc}: {module_name} - NO INSTALADO")
            missing_deps.append(module_name)
            all_ok = False
    
    # 5. Verificar información del modelo
    print("\n🤖 5. INFORMACIÓN DEL MODELO")
    print("-" * 80)
    
    try:
        import json
        with open(models_path / "training_results.json", 'r') as f:
            results = json.load(f)
        
        print(f"   📅 Fecha de entrenamiento: {results['timestamp']}")
        print(f"   🏆 Mejor modelo: {results['best_model']['name']}")
        print(f"   🎯 Accuracy: {results['best_model']['accuracy']:.4f}")
        print(f"   📊 F1-Score: {results['best_model']['f1_score']:.4f}")
        
        # Mostrar información de todos los modelos
        print(f"\n   Modelos entrenados:")
        for model_name, model_info in results['models'].items():
            print(f"      - {model_name}: F1={model_info['test_f1_score']:.4f}, "
                  f"Accuracy={model_info['test_accuracy']:.4f}, "
                  f"Tiempo={model_info['training_time']/60:.2f} min")
        
    except Exception as e:
        print(f"   ⚠️  No se pudo leer información del modelo: {e}")
    
    # 6. Resumen final
    print("\n" + "=" * 80)
    if all_ok and not missing_deps:
        print("✅ ¡TODO LISTO PARA EJECUTAR LA INTERFAZ GRÁFICA!")
        print("\n🚀 Para iniciar la aplicación, ejecuta:")
        print("   cd Entrega3/real_time")
        print("   streamlit run real_time_app.py")
        print("\n📖 O consulta el archivo INICIO_RAPIDO.md para más detalles")
        return 0
    else:
        print("❌ HAY PROBLEMAS QUE RESOLVER:")
        
        if not all_ok:
            print("\n   1. Archivos faltantes:")
            print("      Ejecuta los scripts de Entrega2 en orden:")
            print("      - python 04_model_training_gridsearch.py")
        
        if missing_deps:
            print(f"\n   2. Dependencias faltantes: {', '.join(missing_deps)}")
            print("      Instala con: pip install streamlit opencv-python mediapipe plotly")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
