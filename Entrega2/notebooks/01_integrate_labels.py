"""
Script para integrar las etiquetas de Label Studio con los datos de features
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    LABEL_MAPPING, MAIN_LABELS, EXCLUDED_LABELS,
    FEATURES_PATH, VIDEOS_JSON_PATH, OUTPUT_PATH, PERSONS,
    OTHER_GROUP_VIDEOS, VIDEOS_JSON_PATH_OTHER_GROUP, FEATURES_PATH_OTHER_GROUP
)


def load_json_annotation(json_path):
    """
    Carga un archivo JSON de Label Studio y extrae las anotaciones
    
    Returns:
        List of tuples: [(start_frame, end_frame, label), ...]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Manejar formato de lista o diccionario único
    if isinstance(data, list):
        data = data[0]
    
    annotations = []
    
    if 'annotations' in data and len(data['annotations']) > 0:
        results = data['annotations'][0]['result']
        
        for result in results:
            if result['type'] == 'timelinelabels':
                ranges = result['value']['ranges']
                labels = result['value']['timelinelabels']
                
                for range_item in ranges:
                    start = max(1, range_item['start'])  # Asegurar que sea al menos 1
                    end = range_item['end']
                    
                    for label in labels:
                        annotations.append((start, end, label))
    
    return sorted(annotations, key=lambda x: x[0])


def assign_labels_to_frames(features_df, annotations, person, video_id):
    """
    Asigna etiquetas a cada frame del dataframe de features
    
    Args:
        features_df: DataFrame con las features
        annotations: Lista de tuplas (start, end, label)
        person: Nombre de la persona
        video_id: ID del video
    
    Returns:
        DataFrame con columnas adicionales: label, label_raw, person, video_id, video_speed
    """
    # Crear copia del dataframe
    df = features_df.copy()
    
    # Inicializar columnas
    df['label_raw'] = None
    df['label'] = None
    
    # Asignar etiquetas a cada frame
    for start, end, label_raw in annotations:
        # Encontrar frames en el rango
        mask = (df['frame'] >= start) & (df['frame'] <= end)
        
        # Asignar etiqueta raw
        df.loc[mask, 'label_raw'] = label_raw
        
        # Asignar etiqueta estandarizada
        if label_raw in LABEL_MAPPING:
            df.loc[mask, 'label'] = LABEL_MAPPING[label_raw]
    
    # Agregar metadatos
    df['person'] = person
    df['video_id'] = video_id
    
    # Determinar velocidad del video
    if isinstance(video_id, str) and video_id.startswith('Lento'):
        df['video_speed'] = 'lento'
    else:
        df['video_speed'] = 'normal'
    
    return df


def process_all_videos():
    """
    Procesa todos los videos y genera el dataset integrado
    """
    all_data = []
    stats = {
        'total_videos': 0,
        'total_frames': 0,
        'labeled_frames': 0,
        'excluded_frames': 0,
        'main_label_frames': 0,
        'videos_by_person': {},
        'frames_by_label': {}
    }
    
    features_path = Path(FEATURES_PATH)
    videos_json_path = Path(VIDEOS_JSON_PATH)
    
    print("🔄 Procesando videos y etiquetas...\n")
    
    for person, videos_info in PERSONS.items():
        print(f"📹 Procesando: {person}")
        stats['videos_by_person'][person] = 0
        
        # Procesar videos normales
        for video_num in tqdm(videos_info['normal'], desc=f"  Videos normales"):
            # Construir rutas
            feature_file = features_path / f"{video_num}{person}.csv"
            
            # Determinar nombre del archivo JSON
            if person == "Joshua":
                json_file = videos_json_path / person / f"{video_num}Josh.json"
            else:
                json_file = videos_json_path / person / f"{video_num}{person}.json"
            
            # Verificar que existan los archivos
            if not feature_file.exists():
                print(f"  ⚠️  No se encontró: {feature_file}")
                continue
            if not json_file.exists():
                print(f"  ⚠️  No se encontró: {json_file}")
                continue
            
            # Cargar features
            features_df = pd.read_csv(feature_file)
            
            # Cargar anotaciones
            annotations = load_json_annotation(json_file)
            
            # Asignar etiquetas
            labeled_df = assign_labels_to_frames(
                features_df, annotations, person, str(video_num)
            )
            
            all_data.append(labeled_df)
            stats['total_videos'] += 1
            stats['videos_by_person'][person] += 1
        
        # Procesar videos lentos
        for lento_num in tqdm(videos_info['lento'], desc=f"  Videos lentos"):
            # Construir rutas
            feature_file = features_path / f"Lento{lento_num}.csv"
            json_file = videos_json_path / person / f"Lento{lento_num}.json"
            
            # Verificar que existan los archivos
            if not feature_file.exists():
                print(f"  ⚠️  No se encontró: {feature_file}")
                continue
            if not json_file.exists():
                print(f"  ⚠️  No se encontró: {json_file}")
                continue
            
            # Cargar features
            features_df = pd.read_csv(feature_file)
            
            # Cargar anotaciones
            annotations = load_json_annotation(json_file)
            
            # Asignar etiquetas
            labeled_df = assign_labels_to_frames(
                features_df, annotations, person, f"Lento{lento_num}"
            )
            
            all_data.append(labeled_df)
            stats['total_videos'] += 1
            stats['videos_by_person'][person] += 1
        
        print()
    
    # Consolidar todos los datos
    print("🔗 Consolidando datos...\n")
    df_complete = pd.concat(all_data, ignore_index=True)
    
    # Calcular estadísticas
    stats['total_frames'] = len(df_complete)
    stats['labeled_frames'] = df_complete['label'].notna().sum()
    
    # Frames con etiquetas principales
    df_main = df_complete[df_complete['label'].isin(MAIN_LABELS)]
    stats['main_label_frames'] = len(df_main)
    
    # Frames excluidos
    df_excluded = df_complete[df_complete['label'].isin(EXCLUDED_LABELS)]
    stats['excluded_frames'] = len(df_excluded)
    
    # Distribución por etiqueta
    label_counts = df_complete['label'].value_counts().to_dict()
    # Convertir valores int64 a int para JSON
    stats['frames_by_label'] = {k: int(v) if pd.notna(k) else 'sin_etiqueta' for k, v in label_counts.items()}
    
    # Guardar dataset completo
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)
    
    complete_file = output_path / "labeled_dataset_complete.csv"
    df_complete.to_csv(complete_file, index=False)
    print(f"✅ Dataset completo guardado: {complete_file}")
    
    # Guardar dataset solo con etiquetas principales
    main_file = output_path / "labeled_dataset_main.csv"
    df_main.to_csv(main_file, index=False)
    print(f"✅ Dataset principal guardado: {main_file}")
    
    # Guardar estadísticas (convertir tipos numpy a Python nativos)
    stats_serializable = {
        'total_videos': int(stats['total_videos']),
        'total_frames': int(stats['total_frames']),
        'labeled_frames': int(stats['labeled_frames']),
        'excluded_frames': int(stats['excluded_frames']),
        'main_label_frames': int(stats['main_label_frames']),
        'videos_by_person': {k: int(v) for k, v in stats['videos_by_person'].items()},
        'frames_by_label': stats['frames_by_label']
    }
    
    stats_file = output_path / "integration_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    print(f"✅ Estadísticas guardadas: {stats_file}")
    
    # Guardar mapeo de etiquetas
    mapping_file = output_path / "label_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(LABEL_MAPPING, f, indent=2)
    print(f"✅ Mapeo de etiquetas guardado: {mapping_file}")
    
    return df_complete, df_main, stats


def process_other_group_videos():
    """
    Procesa los videos del otro grupo y retorna el dataframe
    
    IMPORTANTE: Asume que:
    1. Ya ejecutaste split_json_by_video.py para separar los JSON
    2. Ya extrajiste las features de estos videos usando MediaPipe
    """
    all_data = []
    stats = {
        'total_videos': 0,
        'total_frames': 0,
        'labeled_frames': 0,
        'videos_processed': []
    }
    
    videos_json_path = Path(VIDEOS_JSON_PATH_OTHER_GROUP)
    
    # Verificar si el directorio existe
    if not videos_json_path.exists():
        print(f"⚠️  No se encontró el directorio: {videos_json_path}")
        print("   Ejecuta primero split_json_by_video.py para crear los archivos JSON individuales")
        return None, stats
    
    print("\n🔄 Procesando videos del otro grupo...\n")
    
    for person, info in OTHER_GROUP_VIDEOS.items():
        print(f"📹 Procesando: {person}")
        
        for video_name in tqdm(info['videos'], desc=f"  Videos"):
            # Construir rutas
            # NOTA: Las features deberían estar en FEATURES_PATH con el nombre del video
            feature_file = Path(FEATURES_PATH) / f"{video_name}.csv"
            json_file = videos_json_path / f"{video_name}.json"
            
            # Verificar que existan los archivos
            if not json_file.exists():
                print(f"  ⚠️  No se encontró JSON: {json_file}")
                continue
            
            if not feature_file.exists():
                print(f"  ⚠️  No se encontró features CSV: {feature_file}")
                print(f"     Necesitas extraer features de este video primero")
                continue
            
            # Cargar features
            features_df = pd.read_csv(feature_file)
            
            # Cargar anotaciones
            annotations = load_json_annotation(json_file)
            
            # Asignar etiquetas
            labeled_df = assign_labels_to_frames(
                features_df, annotations, person, video_name
            )
            
            all_data.append(labeled_df)
            stats['total_videos'] += 1
            stats['videos_processed'].append(video_name)
        
        print()
    
    if not all_data:
        print("⚠️  No se procesó ningún video del otro grupo")
        return None, stats
    
    # Consolidar datos
    df_other = pd.concat(all_data, ignore_index=True)
    stats['total_frames'] = len(df_other)
    stats['labeled_frames'] = df_other['label'].notna().sum()
    
    return df_other, stats


def print_statistics(stats):
    """
    Imprime estadísticas de forma legible
    """
    print("\n" + "="*60)
    print("📊 ESTADÍSTICAS DE INTEGRACIÓN")
    print("="*60)
    
    print(f"\n📹 Videos procesados: {stats['total_videos']}")
    print(f"   Por persona:")
    for person, count in stats['videos_by_person'].items():
        print(f"   - {person}: {count} videos")
    
    print(f"\n🎞️  Frames totales: {stats['total_frames']:,}")
    print(f"   Frames etiquetados: {stats['labeled_frames']:,} ({stats['labeled_frames']/stats['total_frames']*100:.1f}%)")
    print(f"   Frames sin etiqueta: {stats['total_frames'] - stats['labeled_frames']:,}")
    
    print(f"\n🎯 Frames con etiquetas principales: {stats['main_label_frames']:,}")
    print(f"   Frames excluidos: {stats['excluded_frames']:,}")
    
    print(f"\n🏷️  Distribución por etiqueta:")
    for label, count in sorted(stats['frames_by_label'].items(), key=lambda x: x[1], reverse=True):
        percentage = count / stats['labeled_frames'] * 100
        status = "✓" if label in MAIN_LABELS else "✗"
        print(f"   {status} {label}: {count:,} ({percentage:.1f}%)")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    print("🚀 Iniciando integración de etiquetas\n")
    
    # Procesar videos principales
    df_complete, df_main, stats = process_all_videos()
    
    # Intentar procesar videos del otro grupo
    print("\n" + "="*60)
    print("📦 INTENTANDO INTEGRAR VIDEOS DEL OTRO GRUPO")
    print("="*60)
    
    df_other, stats_other = process_other_group_videos()
    
    if df_other is not None:
        print(f"\n✅ Videos del otro grupo procesados: {stats_other['total_videos']}")
        print(f"   Frames adicionales: {stats_other['total_frames']:,}")
        
        # Combinar con el dataset principal
        df_complete_combined = pd.concat([df_complete, df_other], ignore_index=True)
        
        # Filtrar solo etiquetas principales para el dataset main
        df_main_combined = df_complete_combined[df_complete_combined['label'].isin(MAIN_LABELS)]
        
        # Actualizar estadísticas
        stats['total_videos'] += stats_other['total_videos']
        stats['total_frames'] += stats_other['total_frames']
        stats['labeled_frames'] = df_complete_combined['label'].notna().sum()
        stats['main_label_frames'] = len(df_main_combined)
        stats['videos_by_person']['OtroGrupo'] = stats_other['total_videos']
        
        # Recalcular distribución de etiquetas
        label_counts = df_complete_combined['label'].value_counts().to_dict()
        stats['frames_by_label'] = {k: int(v) if pd.notna(k) else 'sin_etiqueta' for k, v in label_counts.items()}
        
        # Guardar datasets combinados
        output_path = Path(OUTPUT_PATH)
        
        complete_file = output_path / "labeled_dataset_complete.csv"
        df_complete_combined.to_csv(complete_file, index=False)
        print(f"\n✅ Dataset completo COMBINADO guardado: {complete_file}")
        
        main_file = output_path / "labeled_dataset_main.csv"
        df_main_combined.to_csv(main_file, index=False)
        print(f"✅ Dataset principal COMBINADO guardado: {main_file}")
        
        # Actualizar estadísticas
        stats_serializable = {
            'total_videos': int(stats['total_videos']),
            'total_frames': int(stats['total_frames']),
            'labeled_frames': int(stats['labeled_frames']),
            'excluded_frames': int(stats['excluded_frames']),
            'main_label_frames': int(stats['main_label_frames']),
            'videos_by_person': {k: int(v) for k, v in stats['videos_by_person'].items()},
            'frames_by_label': stats['frames_by_label'],
            'includes_other_group': True,
            'other_group_videos': stats_other['videos_processed']
        }
        
        stats_file = output_path / "integration_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats_serializable, f, indent=2)
        print(f"✅ Estadísticas actualizadas: {stats_file}")
    else:
        print("\n⚠️  No se pudieron procesar videos del otro grupo")
        print("   Se usará solo el dataset original")
    
    print_statistics(stats)
    
    print("✅ Integración completada exitosamente!")
    print(f"\n📁 Archivos generados en: {OUTPUT_PATH}/")
    print("   - labeled_dataset_complete.csv (todos los datos)")
    print("   - labeled_dataset_main.csv (solo etiquetas principales)")
    print("   - integration_statistics.json")
    print("   - label_mapping.json")

