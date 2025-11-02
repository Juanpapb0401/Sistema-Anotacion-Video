#!/usr/bin/env python3
"""
Script de inicio rápido para ejecutar todo el pipeline de la Entrega 2
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """Ejecuta un script de Python y maneja errores"""
    print("\n" + "="*80)
    print(f"🚀 {description}")
    print("="*80 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path(__file__).parent,
            check=True,
            text=True
        )
        print(f"\n✅ {description} - COMPLETADO\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en {description}")
        print(f"Código de error: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ No se encontró el script: {script_name}")
        return False


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                    PIPELINE ENTREGA 2 - INICIO RÁPIDO                   ║
    ║              Sistema de Anotación de Video - APO III                    ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    scripts = [
        ("01_integrate_labels.py", "Paso 1: Integración de Etiquetas"),
        ("02_eda_labeled.py", "Paso 2: Análisis Exploratorio de Datos"),
        ("03_data_preparation.py", "Paso 3: Preparación de Datos"),
        ("04_model_training.py", "Paso 4: Entrenamiento de Modelos"),
        ("05_evaluation.py", "Paso 5: Evaluación de Modelos"),
    ]
    
    success_count = 0
    total_scripts = len(scripts)
    
    for script, description in scripts:
        if run_script(script, description):
            success_count += 1
        else:
            print(f"\n⚠️  Pipeline interrumpido en: {description}")
            print("Por favor, revisa los errores antes de continuar.\n")
            sys.exit(1)
    
    print("\n" + "="*80)
    print(f"✅ PIPELINE COMPLETADO: {success_count}/{total_scripts} scripts ejecutados con éxito")
    print("="*80)
    
    print("""
    📁 Archivos generados:
       - data/labeled_dataset_complete.csv
       - data/labeled_dataset_main.csv
       - data/train.csv, validation.csv, test.csv
       - data/integration_statistics.json
       - data/label_mapping.json
       - data/preparation_info.json
       - data/EDA_report.txt
       - data/preparation_report.txt
       - data/evaluation_report.txt
       - data/evaluation_results.json
       - models/*.pkl (todos los modelos)
       - reports/figures/*.png (todas las visualizaciones)
    
    📊 Pipeline completado exitosamente!
       ✅ Datos integrados y preparados
       ✅ Análisis exploratorio realizado
       ✅ Modelos entrenados y optimizados
       ✅ Evaluación completa en conjunto de test
       ✅ Mejor modelo seleccionado y guardado
    
    🎯 Próximos pasos:
       1. Revisar reportes de evaluación
       2. Analizar matrices de confusión
       3. Preparar documento final de Entrega 2
       4. Comenzar con el plan de despliegue (Entrega 3)
    """)


if __name__ == "__main__":
    main()
