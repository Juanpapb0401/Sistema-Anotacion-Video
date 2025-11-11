"""
Script para medir y analizar la performance del sistema en tiempo real
Mide FPS, latencia, uso de recursos y estabilidad de predicciones

Este análisis es ESPECÍFICO para Entrega 3 - Sistema de Tiempo Real
"""

import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import psutil
import sys
from pathlib import Path
from datetime import datetime
from collections import deque, Counter

# Añadir path del sistema de tiempo real
sys.path.append(str(Path(__file__).parent.parent / "real_time"))

from activity_classifier import ActivityClassifier
from video_processor import VideoProcessor

# Configuración
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

OUTPUT_PATH = Path("../reports/figures")
OUTPUT_DATA = Path("../data")


class PerformanceMonitor:
    """Monitor de performance para el sistema de tiempo real"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.fps_history = deque(maxlen=window_size)
        self.latency_history = deque(maxlen=window_size)
        self.cpu_history = deque(maxlen=window_size)
        self.memory_history = deque(maxlen=window_size)
        self.prediction_history = []
        self.confidence_history = []
        self.frame_times = []
        
        self.process = psutil.Process()
    
    def update(self, fps, latency, prediction, confidence):
        """Actualiza las métricas"""
        self.fps_history.append(fps)
        self.latency_history.append(latency)
        self.cpu_history.append(self.process.cpu_percent())
        self.memory_history.append(self.process.memory_info().rss / 1024 / 1024)  # MB
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        self.frame_times.append(time.time())
    
    def get_statistics(self):
        """Calcula estadísticas de performance"""
        return {
            'fps': {
                'mean': np.mean(self.fps_history),
                'std': np.std(self.fps_history),
                'min': np.min(self.fps_history),
                'max': np.max(self.fps_history),
                'p50': np.percentile(self.fps_history, 50),
                'p95': np.percentile(self.fps_history, 95),
            },
            'latency_ms': {
                'mean': np.mean(self.latency_history) * 1000,
                'std': np.std(self.latency_history) * 1000,
                'min': np.min(self.latency_history) * 1000,
                'max': np.max(self.latency_history) * 1000,
                'p50': np.percentile(self.latency_history, 50) * 1000,
                'p95': np.percentile(self.latency_history, 95) * 1000,
            },
            'cpu_percent': {
                'mean': np.mean(self.cpu_history),
                'std': np.std(self.cpu_history),
                'min': np.min(self.cpu_history),
                'max': np.max(self.cpu_history),
            },
            'memory_mb': {
                'mean': np.mean(self.memory_history),
                'std': np.std(self.memory_history),
                'min': np.min(self.memory_history),
                'max': np.max(self.memory_history),
            },
            'predictions': {
                'total': len(self.prediction_history),
                'unique_activities': len(set(self.prediction_history)),
                'most_common': Counter(self.prediction_history).most_common(5),
                'prediction_stability': self._calculate_stability()
            },
            'confidence': {
                'mean': np.mean(self.confidence_history),
                'std': np.std(self.confidence_history),
                'min': np.min(self.confidence_history),
                'max': np.max(self.confidence_history),
            }
        }
    
    def _calculate_stability(self):
        """Calcula la estabilidad de las predicciones (menos cambios = más estable)"""
        if len(self.prediction_history) < 2:
            return 1.0
        
        changes = sum(1 for i in range(1, len(self.prediction_history)) 
                     if self.prediction_history[i] != self.prediction_history[i-1])
        
        stability_score = 1 - (changes / len(self.prediction_history))
        return stability_score


def test_realtime_performance(duration_seconds=30, source=0):
    """
    Prueba la performance del sistema en tiempo real
    
    Args:
        duration_seconds: Duración de la prueba en segundos
        source: Fuente de video (0 para webcam, o ruta a archivo)
    """
    print("=" * 80)
    print(" PRUEBA DE PERFORMANCE EN TIEMPO REAL")
    print("=" * 80)
    print(f"\nDuración de prueba: {duration_seconds} segundos")
    print(f"Fuente: {'Webcam' if source == 0 else source}")
    print("\nPresiona 'q' para terminar antes de tiempo")
    print("\n⏱️  Iniciando en 3 segundos...")
    time.sleep(3)
    
    # Inicializar sistema
    print("\n📦 Cargando modelos...")
    classifier = ActivityClassifier()
    video_processor = VideoProcessor()
    
    # Abrir video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Error: No se puede abrir la fuente de video")
        return None
    
    # Obtener información del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f" Video abierto: {width}x{height}")
    
    # Inicializar monitor
    monitor = PerformanceMonitor(window_size=30)
    
    print("\n🎬 Iniciando captura...")
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            # Verificar tiempo límite
            elapsed = time.time() - start_time
            if elapsed > duration_seconds:
                break
            
            frame_start = time.time()
            
            # Leer frame
            ret, frame = cap.read()
            if not ret:
                print("\n⚠️  No se pudo leer frame")
                break
            
            frame_count += 1
            
            # Procesar frame
            landmarks = video_processor.extract_landmarks(frame)
            
            if landmarks is not None:
                # Extraer features y predecir
                features = video_processor.extract_features_from_landmarks(landmarks)
                prediction, probabilities = classifier.predict(features)
                confidence = np.max(probabilities)
                
                # Dibujar landmarks
                video_processor.draw_landmarks(frame, landmarks)
                
                # Calcular métricas de este frame
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                
                # Actualizar monitor
                monitor.update(fps, frame_time, prediction, confidence)
                
                # Mostrar información en pantalla
                current_fps = np.mean(list(monitor.fps_history)) if monitor.fps_history else fps
                
                cv2.putText(frame, f"Actividad: {prediction}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confianza: {confidence:.2%}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, f"Tiempo restante: {int(duration_seconds - elapsed)}s", 
                           (10, height - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Barra de progreso
                progress = elapsed / duration_seconds
                bar_width = int(width * progress)
                cv2.rectangle(frame, (0, height - 10), (bar_width, height), (0, 255, 0), -1)
            else:
                cv2.putText(frame, "No se detectaron landmarks", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Mostrar frame
            cv2.imshow('Performance Test - Sistema de Tiempo Real', frame)
            
            # Verificar si se presionó 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⏹️  Prueba detenida manualmente")
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        actual_duration = time.time() - start_time
        print(f"\n Prueba completada")
        print(f"   Duración real: {actual_duration:.2f} segundos")
        print(f"   Frames procesados: {frame_count}")
        print(f"   FPS promedio: {frame_count/actual_duration:.2f}")
    
    return monitor


def visualize_performance(monitor):
    """Genera visualizaciones de la performance"""
    print("\n Generando visualizaciones...")
    
    stats = monitor.get_statistics()
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. FPS en el tiempo
    ax1 = fig.add_subplot(gs[0, :2])
    fps_data = list(monitor.fps_history)
    ax1.plot(fps_data, linewidth=2, color='#3498db', label='FPS instantáneo')
    ax1.axhline(y=stats['fps']['mean'], color='red', linestyle='--', 
               linewidth=2, label=f"Promedio: {stats['fps']['mean']:.1f} FPS")
    ax1.fill_between(range(len(fps_data)), fps_data, alpha=0.3)
    ax1.set_xlabel('Frame', fontsize=10, fontweight='bold')
    ax1.set_ylabel('FPS', fontsize=10, fontweight='bold')
    ax1.set_title('📈 FPS en el Tiempo', fontsize=12, fontweight='bold', pad=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribución de FPS
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(fps_data, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.axvline(x=stats['fps']['mean'], color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('FPS', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Frecuencia', fontsize=9, fontweight='bold')
    ax2.set_title(' Distribución de FPS', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Latencia en el tiempo
    ax3 = fig.add_subplot(gs[1, :2])
    latency_ms = [l * 1000 for l in monitor.latency_history]
    ax3.plot(latency_ms, linewidth=2, color='#e74c3c', label='Latencia')
    ax3.axhline(y=stats['latency_ms']['mean'], color='green', linestyle='--', 
               linewidth=2, label=f"Promedio: {stats['latency_ms']['mean']:.1f} ms")
    ax3.fill_between(range(len(latency_ms)), latency_ms, alpha=0.3, color='#e74c3c')
    ax3.set_xlabel('Frame', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Latencia (ms)', fontsize=10, fontweight='bold')
    ax3.set_title('⏱️  Latencia por Frame', fontsize=12, fontweight='bold', pad=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Uso de CPU
    ax4 = fig.add_subplot(gs[1, 2])
    cpu_data = list(monitor.cpu_history)
    ax4.plot(cpu_data, linewidth=2, color='#f39c12')
    ax4.axhline(y=stats['cpu_percent']['mean'], color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Frame', fontsize=9, fontweight='bold')
    ax4.set_ylabel('CPU %', fontsize=9, fontweight='bold')
    ax4.set_title('💻 Uso de CPU', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Uso de Memoria
    ax5 = fig.add_subplot(gs[2, 0])
    mem_data = list(monitor.memory_history)
    ax5.plot(mem_data, linewidth=2, color='#9b59b6')
    ax5.axhline(y=stats['memory_mb']['mean'], color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Frame', fontsize=9, fontweight='bold')
    ax5.set_ylabel('Memoria (MB)', fontsize=9, fontweight='bold')
    ax5.set_title('🧠 Uso de Memoria', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Distribución de Confianza
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(monitor.confidence_history, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax6.axvline(x=stats['confidence']['mean'], color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('Confianza', fontsize=9, fontweight='bold')
    ax6.set_ylabel('Frecuencia', fontsize=9, fontweight='bold')
    ax6.set_title(' Distribución de Confianza', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Actividades detectadas
    ax7 = fig.add_subplot(gs[2, 2])
    activity_counts = Counter(monitor.prediction_history)
    activities = list(activity_counts.keys())
    counts = list(activity_counts.values())
    
    colors_bar = plt.cm.Set3(range(len(activities)))
    ax7.barh(activities, counts, color=colors_bar)
    ax7.set_xlabel('Frecuencia', fontsize=9, fontweight='bold')
    ax7.set_title('🏃 Actividades Detectadas', fontsize=11, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(' Análisis de Performance - Sistema de Tiempo Real', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH / "04_realtime_performance_analysis.png", dpi=300, bbox_inches='tight')
    print(f"   💾 Gráfico guardado: 04_realtime_performance_analysis.png")
    plt.close()


def create_performance_summary(stats):
    """Crea una tabla resumen de performance"""
    print("\n📋 Generando tabla resumen...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Preparar datos
    summary_data = [
        ['Métrica', 'Media', 'Min', 'Max', 'P95', 'Evaluación'],
        ['FPS', 
         f"{stats['fps']['mean']:.1f}",
         f"{stats['fps']['min']:.1f}",
         f"{stats['fps']['max']:.1f}",
         f"{stats['fps']['p95']:.1f}",
         ' Excelente' if stats['fps']['mean'] > 25 else '⚠️ Aceptable' if stats['fps']['mean'] > 15 else '❌ Insuficiente'],
        ['Latencia (ms)', 
         f"{stats['latency_ms']['mean']:.1f}",
         f"{stats['latency_ms']['min']:.1f}",
         f"{stats['latency_ms']['max']:.1f}",
         f"{stats['latency_ms']['p95']:.1f}",
         ' Excelente' if stats['latency_ms']['mean'] < 50 else '⚠️ Aceptable' if stats['latency_ms']['mean'] < 100 else '❌ Lento'],
        ['CPU (%)', 
         f"{stats['cpu_percent']['mean']:.1f}",
         f"{stats['cpu_percent']['min']:.1f}",
         f"{stats['cpu_percent']['max']:.1f}",
         '-',
         ' Bajo' if stats['cpu_percent']['mean'] < 40 else '⚠️ Medio' if stats['cpu_percent']['mean'] < 70 else '❌ Alto'],
        ['Memoria (MB)', 
         f"{stats['memory_mb']['mean']:.0f}",
         f"{stats['memory_mb']['min']:.0f}",
         f"{stats['memory_mb']['max']:.0f}",
         '-',
         ' Eficiente' if stats['memory_mb']['mean'] < 500 else '⚠️ Moderado' if stats['memory_mb']['mean'] < 1000 else '❌ Alto'],
        ['Confianza', 
         f"{stats['confidence']['mean']:.2%}",
         f"{stats['confidence']['min']:.2%}",
         f"{stats['confidence']['max']:.2%}",
         '-',
         ' Alta' if stats['confidence']['mean'] > 0.8 else '⚠️ Media' if stats['confidence']['mean'] > 0.6 else '❌ Baja'],
        ['Estabilidad', 
         f"{stats['predictions']['prediction_stability']:.2%}",
         '-', '-', '-',
         ' Muy estable' if stats['predictions']['prediction_stability'] > 0.8 else '⚠️ Moderada' if stats['predictions']['prediction_stability'] > 0.6 else '❌ Inestable'],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                    colWidths=[0.20, 0.12, 0.12, 0.12, 0.12, 0.20])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Estilizar encabezado
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Estilizar filas
    for i in range(1, len(summary_data)):
        for j in range(6):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('white')
            
            # Resaltar columna de evaluación
            if j == 5:
                text = summary_data[i][j]
                if '' in text:
                    cell.set_facecolor('#2ecc71')
                    cell.set_text_props(weight='bold')
                elif '⚠️' in text:
                    cell.set_facecolor('#f39c12')
                    cell.set_text_props(weight='bold')
                elif '❌' in text:
                    cell.set_facecolor('#e74c3c')
                    cell.set_text_props(weight='bold', color='white')
    
    plt.title(' Resumen de Performance - Sistema de Tiempo Real', 
             fontsize=14, fontweight='bold', pad=20)
    plt.savefig(OUTPUT_PATH / "05_performance_summary_table.png", dpi=300, bbox_inches='tight')
    print(f"   💾 Tabla guardada: 05_performance_summary_table.png")
    plt.close()


def generate_performance_report(stats):
    """Genera reporte textual completo"""
    print("\n📝 Generando reporte de performance...")
    
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("REPORTE DE PERFORMANCE EN TIEMPO REAL")
    report_lines.append("Sistema de Clasificación de Actividades - Entrega 3")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 1. Resumen ejecutivo
    report_lines.append("📌 RESUMEN EJECUTIVO")
    report_lines.append("-" * 80)
    report_lines.append(f"FPS promedio:         {stats['fps']['mean']:.1f} FPS")
    report_lines.append(f"Latencia promedio:    {stats['latency_ms']['mean']:.1f} ms")
    report_lines.append(f"CPU promedio:         {stats['cpu_percent']['mean']:.1f}%")
    report_lines.append(f"Memoria promedio:     {stats['memory_mb']['mean']:.0f} MB")
    report_lines.append(f"Confianza promedio:   {stats['confidence']['mean']:.2%}")
    report_lines.append(f"Estabilidad:          {stats['predictions']['prediction_stability']:.2%}")
    report_lines.append("")
    
    # 2. FPS detallado
    report_lines.append("🎬 FRAMES POR SEGUNDO (FPS)")
    report_lines.append("-" * 80)
    report_lines.append(f"Media:       {stats['fps']['mean']:.2f} FPS")
    report_lines.append(f"Mediana:     {stats['fps']['p50']:.2f} FPS")
    report_lines.append(f"Desv. Est.:  {stats['fps']['std']:.2f}")
    report_lines.append(f"Mínimo:      {stats['fps']['min']:.2f} FPS")
    report_lines.append(f"Máximo:      {stats['fps']['max']:.2f} FPS")
    report_lines.append(f"Percentil 95: {stats['fps']['p95']:.2f} FPS")
    
    # Evaluación de FPS
    if stats['fps']['mean'] > 25:
        report_lines.append(" Evaluación: EXCELENTE - Ideal para tiempo real")
    elif stats['fps']['mean'] > 15:
        report_lines.append("⚠️  Evaluación: ACEPTABLE - Funcional pero mejorable")
    else:
        report_lines.append("❌ Evaluación: INSUFICIENTE - Requiere optimización")
    report_lines.append("")
    
    # 3. Latencia
    report_lines.append("⏱️  LATENCIA POR FRAME")
    report_lines.append("-" * 80)
    report_lines.append(f"Media:       {stats['latency_ms']['mean']:.2f} ms")
    report_lines.append(f"Mediana:     {stats['latency_ms']['p50']:.2f} ms")
    report_lines.append(f"Desv. Est.:  {stats['latency_ms']['std']:.2f} ms")
    report_lines.append(f"Mínimo:      {stats['latency_ms']['min']:.2f} ms")
    report_lines.append(f"Máximo:      {stats['latency_ms']['max']:.2f} ms")
    report_lines.append(f"Percentil 95: {stats['latency_ms']['p95']:.2f} ms")
    
    if stats['latency_ms']['mean'] < 50:
        report_lines.append(" Evaluación: EXCELENTE - Muy responsivo")
    elif stats['latency_ms']['mean'] < 100:
        report_lines.append("⚠️  Evaluación: ACEPTABLE - Latencia perceptible")
    else:
        report_lines.append("❌ Evaluación: LENTO - Usuarios notarán retraso")
    report_lines.append("")
    
    # 4. Recursos del sistema
    report_lines.append("💻 USO DE RECURSOS")
    report_lines.append("-" * 80)
    report_lines.append(f"CPU:")
    report_lines.append(f"  Media:   {stats['cpu_percent']['mean']:.1f}%")
    report_lines.append(f"  Mínimo:  {stats['cpu_percent']['min']:.1f}%")
    report_lines.append(f"  Máximo:  {stats['cpu_percent']['max']:.1f}%")
    report_lines.append(f"Memoria:")
    report_lines.append(f"  Media:   {stats['memory_mb']['mean']:.0f} MB")
    report_lines.append(f"  Mínimo:  {stats['memory_mb']['min']:.0f} MB")
    report_lines.append(f"  Máximo:  {stats['memory_mb']['max']:.0f} MB")
    report_lines.append("")
    
    # 5. Predicciones
    report_lines.append("🎯 ANÁLISIS DE PREDICCIONES")
    report_lines.append("-" * 80)
    report_lines.append(f"Total de predicciones:  {stats['predictions']['total']}")
    report_lines.append(f"Actividades únicas:     {stats['predictions']['unique_activities']}")
    report_lines.append(f"Estabilidad:            {stats['predictions']['prediction_stability']:.2%}")
    report_lines.append(f"\nActividades más detectadas:")
    for activity, count in stats['predictions']['most_common']:
        pct = (count / stats['predictions']['total']) * 100
        report_lines.append(f"  {activity:<25} {count:>5} ({pct:>5.1f}%)")
    report_lines.append("")
    
    # 6. Confianza
    report_lines.append(" CONFIANZA DE PREDICCIONES")
    report_lines.append("-" * 80)
    report_lines.append(f"Media:   {stats['confidence']['mean']:.2%}")
    report_lines.append(f"Mínimo:  {stats['confidence']['min']:.2%}")
    report_lines.append(f"Máximo:  {stats['confidence']['max']:.2%}")
    
    if stats['confidence']['mean'] > 0.8:
        report_lines.append(" Evaluación: ALTA - Predicciones muy confiables")
    elif stats['confidence']['mean'] > 0.6:
        report_lines.append("⚠️  Evaluación: MEDIA - Confianza moderada")
    else:
        report_lines.append("❌ Evaluación: BAJA - Revisar modelo o datos")
    report_lines.append("")
    
    # 7. Conclusiones
    report_lines.append("🎯 CONCLUSIONES")
    report_lines.append("-" * 80)
    
    # Determinar conclusiones basadas en métricas
    if stats['fps']['mean'] > 20 and stats['latency_ms']['mean'] < 70 and stats['confidence']['mean'] > 0.7:
        report_lines.append(" El sistema es APTO para producción en tiempo real")
        report_lines.append("   - Performance fluido y responsivo")
        report_lines.append("   - Predicciones confiables")
        report_lines.append("   - Uso eficiente de recursos")
    else:
        report_lines.append("⚠️  El sistema requiere OPTIMIZACIONES antes de producción")
        if stats['fps']['mean'] <= 20:
            report_lines.append("   - Mejorar FPS (actualmente bajo)")
        if stats['latency_ms']['mean'] >= 70:
            report_lines.append("   - Reducir latencia (actualmente alta)")
        if stats['confidence']['mean'] <= 0.7:
            report_lines.append("   - Mejorar confianza de predicciones")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Guardar reporte
    report_text = "\n".join(report_lines)
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DATA / "realtime_performance_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"   💾 Reporte guardado: realtime_performance_report.txt")
    
    # Guardar JSON
    with open(OUTPUT_DATA / "realtime_performance_stats.json", 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"   💾 Estadísticas guardadas: realtime_performance_stats.json")
    
    return report_text


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prueba de performance en tiempo real')
    parser.add_argument('--duration', type=int, default=30, 
                       help='Duración de la prueba en segundos (default: 30)')
    parser.add_argument('--source', type=int, default=0, 
                       help='Fuente de video (0=webcam, o índice) (default: 0)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(" ANÁLISIS DE PERFORMANCE EN TIEMPO REAL")
    print("   Sistema de Clasificación de Actividades Humanas")
    print("=" * 80)
    print()
    
    # 1. Ejecutar prueba
    monitor = test_realtime_performance(
        duration_seconds=args.duration,
        source=args.source
    )
    
    if monitor is None:
        print("\n❌ Error en la prueba de performance")
        return
    
    # 2. Calcular estadísticas
    stats = monitor.get_statistics()
    
    # 3. Generar visualizaciones
    visualize_performance(monitor)
    create_performance_summary(stats)
    
    # 4. Generar reporte
    report = generate_performance_report(stats)
    
    print("\n" + "=" * 80)
    print(report)
    print("=" * 80)
    
    print("\n Análisis de performance completado!")
    print(f"\n📁 Archivos generados:")
    print(f"   Gráficos:")
    print(f"      - {OUTPUT_PATH}/04_realtime_performance_analysis.png")
    print(f"      - {OUTPUT_PATH}/05_performance_summary_table.png")
    print(f"   Datos:")
    print(f"      - {OUTPUT_DATA}/realtime_performance_report.txt")
    print(f"      - {OUTPUT_DATA}/realtime_performance_stats.json")


if __name__ == "__main__":
    main()
