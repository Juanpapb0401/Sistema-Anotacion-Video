#!/usr/bin/env python3
"""
Script para verificar que los arreglos están correctos
"""

import sys
from pathlib import Path

# Añadir el directorio real_time al path
sys.path.insert(0, str(Path(__file__).parent / 'real_time'))

def test_imports():
    """Verifica que los imports funcionan"""
    print("🔍 Verificando imports...")
    try:
        from activity_classifier import ActivityClassifier
        from video_processor import VideoProcessor
        print("✅ Imports correctos")
        return True
    except Exception as e:
        print(f"❌ Error en imports: {e}")
        return False

def test_classifier_statistics():
    """Verifica que get_statistics retorna las keys correctas"""
    print("\n🔍 Verificando estructura de estadísticas...")
    try:
        from activity_classifier import ActivityClassifier
        
        # Paths de los modelos
        model_path = Path(__file__).parent.parent / 'Entrega2' / 'models' / 'best_model.pkl'
        scaler_path = Path(__file__).parent.parent / 'Entrega2' / 'models' / 'scaler.pkl'
        label_encoder_path = Path(__file__).parent.parent / 'Entrega2' / 'models' / 'label_encoder.pkl'
        
        if not all([model_path.exists(), scaler_path.exists(), label_encoder_path.exists()]):
            print("⚠️ Modelos no encontrados, saltando test")
            return True
        
        # Crear clasificador
        classifier = ActivityClassifier(
            str(model_path),
            str(scaler_path),
            str(label_encoder_path)
        )
        
        # Obtener estadísticas
        stats = classifier.get_statistics()
        
        # Verificar keys
        required_keys = ['total_predictions', 'class_distribution', 'class_percentages']
        for key in required_keys:
            if key not in stats:
                print(f"❌ Falta key '{key}' en estadísticas")
                return False
        
        print(f"✅ Estructura correcta: {list(stats.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_syntax():
    """Verifica que no hay errores de sintaxis en real_time_app.py"""
    print("\n🔍 Verificando sintaxis de real_time_app.py...")
    try:
        app_file = Path(__file__).parent / 'real_time' / 'real_time_app.py'
        with open(app_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        compile(code, str(app_file), 'exec')
        print("✅ Sintaxis correcta")
        return True
    except SyntaxError as e:
        print(f"❌ Error de sintaxis en línea {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("TEST DE ARREGLOS - Sistema de Clasificación en Tiempo Real")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Sintaxis", test_syntax),
        ("Estadísticas", test_classifier_statistics),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("RESUMEN DE TESTS")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 Todos los tests pasaron!")
        print("\nPuedes ejecutar la aplicación con:")
        print("  cd Entrega3")
        print("  ./run_app.sh")
    else:
        print("⚠️ Algunos tests fallaron. Revisa los errores arriba.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
