"""
Análisis Exploratorio de Datos (EDA) para el dataset etiquetado
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Rutas
DATA_PATH = "../data"
FIGURES_PATH = "../reports/figures"


def load_data():
    """Carga el dataset etiquetado"""
    data_path = Path(DATA_PATH)
    
    df_complete = pd.read_csv(data_path / "labeled_dataset_complete.csv")
    df_main = pd.read_csv(data_path / "labeled_dataset_main.csv")
    
    with open(data_path / "integration_statistics.json", 'r') as f:
        stats = json.load(f)
    
    return df_complete, df_main, stats


def plot_class_distribution(df, save_path):
    """Gráfico de distribución de clases"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Conteo absoluto
    class_counts = df['label'].value_counts()
    class_counts.plot(kind='bar', ax=ax1, color='steelblue')
    ax1.set_title('Distribución de Clases (Valores Absolutos)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Clase', fontsize=12)
    ax1.set_ylabel('Número de Frames', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Añadir valores en las barras
    for i, v in enumerate(class_counts):
        ax1.text(i, v + 100, str(v), ha='center', fontweight='bold')
    
    # Porcentajes
    class_pct = df['label'].value_counts(normalize=True) * 100
    class_pct.plot(kind='bar', ax=ax2, color='coral')
    ax2.set_title('Distribución de Clases (Porcentajes)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Clase', fontsize=12)
    ax2.set_ylabel('Porcentaje (%)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # Añadir valores en las barras
    for i, v in enumerate(class_pct):
        ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / '01_class_distribution.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico guardado: 01_class_distribution.png")
    plt.close()


def plot_distribution_by_person(df, save_path):
    """Distribución de clases por persona"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    persons = df['person'].unique()
    
    for idx, person in enumerate(persons):
        df_person = df[df['person'] == person]
        class_counts = df_person['label'].value_counts()
        
        class_counts.plot(kind='bar', ax=axes[idx], color=f'C{idx}')
        axes[idx].set_title(f'{person} - {len(df_person):,} frames', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Clase', fontsize=10)
        axes[idx].set_ylabel('Número de Frames', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Añadir valores
        for i, v in enumerate(class_counts):
            axes[idx].text(i, v + 50, str(v), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path / '02_distribution_by_person.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico guardado: 02_distribution_by_person.png")
    plt.close()


def plot_speed_comparison(df, save_path):
    """Comparación entre videos normales y lentos"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribución por velocidad
    speed_counts = df.groupby(['video_speed', 'label']).size().unstack(fill_value=0)
    speed_counts.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Distribución de Clases por Velocidad de Video', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Velocidad', fontsize=12)
    ax1.set_ylabel('Número de Frames', fontsize=12)
    ax1.legend(title='Clase', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.tick_params(axis='x', rotation=0)
    
    # Proporción de frames por velocidad
    speed_pct = df['video_speed'].value_counts()
    colors = ['#3498db', '#e74c3c']
    ax2.pie(speed_pct, labels=speed_pct.index, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Proporción de Frames por Velocidad', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / '03_speed_comparison.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico guardado: 03_speed_comparison.png")
    plt.close()


def plot_feature_distributions(df, save_path):
    """Distribución de características por clase"""
    features = ['left_knee_angle', 'right_knee_angle', 'left_elbow_angle', 'right_elbow_angle']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        for label in df['label'].unique():
            df_label = df[df['label'] == label]
            axes[idx].hist(df_label[feature], bins=30, alpha=0.5, label=label)
        
        axes[idx].set_title(f'Distribución: {feature}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Ángulo (grados)', fontsize=10)
        axes[idx].set_ylabel('Frecuencia', fontsize=10)
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / '04_feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico guardado: 04_feature_distributions.png")
    plt.close()


def plot_feature_boxplots(df, save_path):
    """Box plots de características por clase"""
    features = ['left_knee_angle', 'right_knee_angle', 'left_elbow_angle', 'right_elbow_angle']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        df.boxplot(column=feature, by='label', ax=axes[idx])
        axes[idx].set_title(f'{feature}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Clase', fontsize=10)
        axes[idx].set_ylabel('Ángulo (grados)', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)
        plt.sca(axes[idx])
        plt.xticks(rotation=45, ha='right')
    
    plt.suptitle('Box Plots de Características por Clase', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path / '05_feature_boxplots.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico guardado: 05_feature_boxplots.png")
    plt.close()


def plot_correlation_matrix(df, save_path):
    """Matriz de correlación de características"""
    # Seleccionar solo columnas numéricas de features
    feature_cols = [col for col in df.columns if col not in ['frame', 'label_raw', 'label', 'person', 'video_id', 'video_speed']]
    
    corr_matrix = df[feature_cols].corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Matriz de Correlación de Características', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path / '06_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico guardado: 06_correlation_matrix.png")
    plt.close()


def plot_temporal_analysis(df, save_path):
    """Análisis temporal de duración de actividades"""
    # Calcular duración de cada segmento de actividad
    durations = []
    
    for person in df['person'].unique():
        for video_id in df[df['person'] == person]['video_id'].unique():
            df_video = df[(df['person'] == person) & (df['video_id'] == video_id)]
            
            # Detectar cambios de etiqueta
            df_video = df_video.sort_values('frame')
            df_video['label_change'] = (df_video['label'] != df_video['label'].shift()).cumsum()
            
            # Agrupar por segmento
            segments = df_video.groupby('label_change').agg({
                'frame': ['min', 'max'],
                'label': 'first'
            }).reset_index(drop=True)
            
            segments.columns = ['frame_start', 'frame_end', 'label']
            segments['duration'] = segments['frame_end'] - segments['frame_start'] + 1
            
            durations.append(segments[['label', 'duration']])
    
    df_durations = pd.concat(durations, ignore_index=True)
    
    # Gráfico
    plt.figure(figsize=(14, 6))
    df_durations.boxplot(column='duration', by='label', figsize=(14, 6))
    plt.title('Duración de Actividades (en frames)', fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remover título automático
    plt.xlabel('Clase', fontsize=12)
    plt.ylabel('Duración (frames)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path / '07_temporal_analysis.png', dpi=300, bbox_inches='tight')
    print(f" Gráfico guardado: 07_temporal_analysis.png")
    plt.close()
    
    return df_durations


def generate_summary_report(df_complete, df_main, stats, df_durations, save_path):
    """Genera un reporte de texto con estadísticas"""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("REPORTE DE ANÁLISIS EXPLORATORIO DE DATOS")
    report_lines.append("Sistema de Anotación de Video - Entrega 2")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Información general
    report_lines.append(" INFORMACIÓN GENERAL")
    report_lines.append("-" * 80)
    report_lines.append(f"Total de videos procesados: {stats['total_videos']}")
    report_lines.append(f"Total de frames: {stats['total_frames']:,}")
    report_lines.append(f"Frames etiquetados: {stats['labeled_frames']:,} ({stats['labeled_frames']/stats['total_frames']*100:.1f}%)")
    report_lines.append(f"Frames con etiquetas principales: {stats['main_label_frames']:,}")
    report_lines.append("")
    
    # Distribución por persona
    report_lines.append(" VIDEOS POR PERSONA")
    report_lines.append("-" * 80)
    for person, count in stats['videos_by_person'].items():
        person_frames = len(df_main[df_main['person'] == person])
        report_lines.append(f"{person}: {count} videos, {person_frames:,} frames")
    report_lines.append("")
    
    # Distribución de clases
    report_lines.append("🏷️  DISTRIBUCIÓN DE CLASES (PRINCIPALES)")
    report_lines.append("-" * 80)
    class_counts = df_main['label'].value_counts()
    for label, count in class_counts.items():
        pct = count / len(df_main) * 100
        report_lines.append(f"{label}: {count:,} frames ({pct:.2f}%)")
    report_lines.append("")
    
    # Balance de clases
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count
    report_lines.append("️  BALANCE DE CLASES")
    report_lines.append("-" * 80)
    report_lines.append(f"Clase más frecuente: {class_counts.idxmax()} ({max_count:,} frames)")
    report_lines.append(f"Clase menos frecuente: {class_counts.idxmin()} ({min_count:,} frames)")
    report_lines.append(f"Ratio de desbalance: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 3:
        report_lines.append("️  ADVERTENCIA: Dataset desbalanceado. Considerar técnicas de balanceo.")
    report_lines.append("")
    
    # Duración de actividades
    report_lines.append("️  DURACIÓN PROMEDIO DE ACTIVIDADES (frames)")
    report_lines.append("-" * 80)
    avg_durations = df_durations.groupby('label')['duration'].agg(['mean', 'std', 'min', 'max'])
    for label in avg_durations.index:
        stats_row = avg_durations.loc[label]
        report_lines.append(f"{label}:")
        report_lines.append(f"  Media: {stats_row['mean']:.1f} ± {stats_row['std']:.1f}")
        report_lines.append(f"  Rango: [{stats_row['min']:.0f}, {stats_row['max']:.0f}]")
    report_lines.append("")
    
    # Velocidad de videos
    report_lines.append("🎥 DISTRIBUCIÓN POR VELOCIDAD")
    report_lines.append("-" * 80)
    speed_counts = df_main['video_speed'].value_counts()
    for speed, count in speed_counts.items():
        pct = count / len(df_main) * 100
        report_lines.append(f"{speed}: {count:,} frames ({pct:.2f}%)")
    report_lines.append("")
    
    # Estadísticas de características
    report_lines.append("📐 ESTADÍSTICAS DE CARACTERÍSTICAS")
    report_lines.append("-" * 80)
    feature_cols = ['left_knee_angle', 'right_knee_angle', 'left_elbow_angle', 'right_elbow_angle']
    for feature in feature_cols:
        report_lines.append(f"\n{feature}:")
        stats_by_class = df_main.groupby('label')[feature].agg(['mean', 'std'])
        for label in stats_by_class.index:
            mean_val = stats_by_class.loc[label, 'mean']
            std_val = stats_by_class.loc[label, 'std']
            report_lines.append(f"  {label}: {mean_val:.2f}° ± {std_val:.2f}°")
    report_lines.append("")
    
    # Recomendaciones
    report_lines.append("💡 RECOMENDACIONES")
    report_lines.append("-" * 80)
    
    if imbalance_ratio > 3:
        report_lines.append("1. Aplicar técnicas de balanceo de clases:")
        report_lines.append("   - SMOTE (Synthetic Minority Over-sampling Technique)")
        report_lines.append("   - Oversampling de clases minoritarias")
        report_lines.append("   - Undersampling de clases mayoritarias")
        report_lines.append("   - Class weights en el modelo")
    
    if len(df_main) < 10000:
        report_lines.append("2. Aumentar el dataset:")
        report_lines.append("   - Data augmentation (velocidad, ruido, interpolación)")
        report_lines.append("   - Captura de más videos")
    
    report_lines.append("3. Preprocesamiento adicional:")
    report_lines.append("   - Normalización de características")
    report_lines.append("   - Extracción de características temporales (ventanas)")
    report_lines.append("   - Feature engineering basado en dominio")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Guardar reporte
    report_text = "\n".join(report_lines)
    report_file = save_path / "EDA_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n Reporte guardado: EDA_report.txt")
    
    return report_text


def main():
    """Función principal"""
    print(" Iniciando Análisis Exploratorio de Datos\n")
    
    # Crear carpeta de figuras
    figures_path = Path(FIGURES_PATH)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # Cargar datos
    print(" Cargando datos...")
    df_complete, df_main, stats = load_data()
    print(f"   Dataset completo: {len(df_complete):,} frames")
    print(f"   Dataset principal: {len(df_main):,} frames")
    print()
    
    # Generar gráficos
    print(" Generando gráficos...\n")
    
    plot_class_distribution(df_main, figures_path)
    plot_distribution_by_person(df_main, figures_path)
    plot_speed_comparison(df_main, figures_path)
    plot_feature_distributions(df_main, figures_path)
    plot_feature_boxplots(df_main, figures_path)
    plot_correlation_matrix(df_main, figures_path)
    df_durations = plot_temporal_analysis(df_main, figures_path)
    
    # Generar reporte
    print("\n Generando reporte de texto...")
    report = generate_summary_report(df_complete, df_main, stats, df_durations, Path(DATA_PATH))
    
    print("\n" + "=" * 80)
    print(report)
    
    print("\n Análisis exploratorio completado!")
    print(f" Gráficos guardados en: {FIGURES_PATH}/")
    print(f"📄 Reporte guardado en: {DATA_PATH}/EDA_report.txt")


if __name__ == "__main__":
    main()
