#!/usr/bin/env python3
"""
Script maestro para ejecutar todas las evaluaciones de Entrega 3
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def print_header(text):
    """Imprime un encabezado decorado"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def print_step(number, text):
    """Imprime un paso numerado"""
    print(f"\n{'='*80}")
    print(f"PASO {number}: {text}")
    print('='*80 + "\n")

def run_script(script_path, description):
    """Ejecuta un script de Python"""
    print(f" Ejecutando: {script_path.name}")
    print(f"   Descripción: {description}")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            check=True,
            text=True
        )
        print(f"\n {script_path.name} completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error ejecutando {script_path.name}")
        print(f"   Código de salida: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        return False

def main():
    """Función principal"""
    print_header("🎯 EVALUACIÓN COMPLETA - ENTREGA 3")
    print("Sistema de Clasificación de Actividades Humanas")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verificar ubicación
    current_dir = Path(__file__).parent
    print(f"\n📂 Directorio de trabajo: {current_dir}")
    
    # Lista de scripts a ejecutar
    scripts = [
        {
            'path': current_dir / 'evaluation' / '01_analyze_feature_reduction.py',
            'description': 'Análisis de reducción de características (147 → 15)',
            'optional': False
        },
        {
            'path': current_dir / 'evaluation' / '02_test_realtime_performance.py',
            'description': 'Prueba de performance en tiempo real (requiere webcam)',
            'optional': True
        }
    ]
    
    # Verificar que los scripts existen
    print("\n📋 Verificando scripts...")
    missing = []
    for script_info in scripts:
        if script_info['path'].exists():
            print(f"    {script_info['path'].name}")
        else:
            print(f"   ❌ {script_info['path'].name} - NO ENCONTRADO")
            missing.append(script_info['path'].name)
    
    if missing:
        print(f"\n❌ Error: Faltan scripts necesarios: {', '.join(missing)}")
        return False
    
    print("\n Todos los scripts están disponibles")
    
    # Ejecutar scripts
    results = []
    
    for i, script_info in enumerate(scripts, 1):
        print_step(i, script_info['description'].upper())
        
        if script_info['optional']:
            response = input("\n⚠️  Este script requiere webcam. ¿Ejecutar? (s/n): ").lower()
            if response != 's':
                print("⏭️  Saltando este script...")
                results.append({'script': script_info['path'].name, 'status': 'skipped'})
                continue
        
        success = run_script(script_info['path'], script_info['description'])
        results.append({
            'script': script_info['path'].name,
            'status': 'success' if success else 'failed'
        })
        
        if not success and not script_info['optional']:
            print("\n❌ Script crítico falló. Abortando ejecución...")
            break
    
    # Resumen final
    print_header(" RESUMEN DE EVALUACIÓN")
    
    for result in results:
        status_emoji = {
            'success': '',
            'failed': '❌',
            'skipped': '⏭️'
        }.get(result['status'], '❓')
        
        status_text = {
            'success': 'EXITOSO',
            'failed': 'FALLÓ',
            'skipped': 'SALTADO'
        }.get(result['status'], 'DESCONOCIDO')
        
        print(f"  {status_emoji} {result['script']:<45} [{status_text}]")
    
    # Verificar outputs generados
    print("\n📁 Verificando archivos generados...")
    
    expected_files = [
        ('Gráficos', [
            current_dir / 'reports' / 'figures' / '01_feature_importance_analysis.png',
            current_dir / 'reports' / 'figures' / '02_feature_types_distribution.png',
            current_dir / 'reports' / 'figures' / '03_dimensionality_impact.png',
        ]),
        ('Datos', [
            current_dir / 'data' / 'feature_reduction_report.txt',
            current_dir / 'data' / 'feature_reduction_analysis.json',
        ])
    ]
    
    for category, files in expected_files:
        print(f"\n   {category}:")
        for file_path in files:
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"       {file_path.name} ({size_kb:.1f} KB)")
            else:
                print(f"      ❌ {file_path.name} - NO GENERADO")
    
    # Mensaje final
    print("\n" + "=" * 80)
    success_count = sum(1 for r in results if r['status'] == 'success')
    total_count = len([r for r in results if r['status'] != 'skipped'])
    
    if success_count == total_count and total_count > 0:
        print("🎉 ¡EVALUACIÓN COMPLETADA CON ÉXITO!")
        print("\n📋 Próximos pasos:")
        print("   1. Revisa los reportes en Entrega3/data/")
        print("   2. Revisa los gráficos en Entrega3/reports/figures/")
        print("   3. Incluye los resultados en INFORME_ENTREGA3.md")
        print("   4. Usa los gráficos en tu presentación de video")
    else:
        print("⚠️  EVALUACIÓN COMPLETADA CON ADVERTENCIAS")
        print(f"   Scripts exitosos: {success_count}/{total_count}")
        print("\n💡 Revisa los mensajes de error arriba para más detalles")
    
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Evaluación interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {str(e)}")
        sys.exit(1)
