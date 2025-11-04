"""
Test rápido para verificar que los modelos se cargan correctamente
"""

import sys
from pathlib import Path

# Agregar path para importar los módulos
sys.path.append(str(Path(__file__).parent / "real_time"))

def test_load_models():
    print("🧪 TEST: Carga de Modelos")
    print("=" * 80)
    
    base_path = Path(__file__).parent.parent / "Entrega2"
    
    model_path = base_path / "models" / "best_model.pkl"
    scaler_path = base_path / "models" / "scaler.pkl"
    label_encoder_path = base_path / "models" / "label_encoder.pkl"
    
    print(f"\n📂 Rutas:")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   Label Encoder: {label_encoder_path}")
    
    print(f"\n✅ Archivos existen:")
    print(f"   Model: {model_path.exists()}")
    print(f"   Scaler: {scaler_path.exists()}")
    print(f"   Label Encoder: {label_encoder_path.exists()}")
    
    if not all([model_path.exists(), scaler_path.exists(), label_encoder_path.exists()]):
        print("\n❌ Error: Faltan archivos")
        return False
    
    print("\n🔄 Intentando cargar VideoProcessor...")
    try:
        from video_processor import VideoProcessor
        video_processor = VideoProcessor()
        print("   ✅ VideoProcessor cargado")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🔄 Intentando cargar ActivityClassifier...")
    try:
        from activity_classifier import ActivityClassifier
        classifier = ActivityClassifier(
            str(model_path),
            str(scaler_path),
            str(label_encoder_path)
        )
        print("   ✅ ActivityClassifier cargado")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✅ ¡TODOS LOS TESTS PASARON!")
    print("\nLa interfaz debería funcionar correctamente.")
    print("Ejecuta: streamlit run real_time/real_time_app.py")
    
    return True


if __name__ == "__main__":
    success = test_load_models()
    sys.exit(0 if success else 1)
