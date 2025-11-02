"""
Script para extraer landmarks (features de MediaPipe) de los videos del otro grupo.

Este script:
1. Lee los videos de raw_videos_otro_grupo/
2. Extrae las coordenadas de pose usando MediaPipe
3. Guarda los CSV en 03_features/ con el nombre del video (ej: VIDEO_01.csv)
"""

import cv2
import mediapipe as mp
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas
INPUT_PATH = "../data/raw_videos_otro_grupo"
OUTPUT_PATH = "../data/03_features"

# Lista de videos esperados del otro grupo (basado en los JSON y videos disponibles)
EXPECTED_VIDEOS = [
    "VIDEO_01", "VIDEO_03", "VIDEO_04", "VIDEO_05", "VIDEO_06",
    "VIDEO_07", "VIDEO_08", "VIDEO_09", "VIDEO_010", "VIDEO_011",
    "VIDEO_012", "VIDEO_013", "VIDEO_014", "VIDEO_015", "VIDEO_016",
    "VIDEO_017", "VIDEO_018", "VIDEO_019", "VIDEO_020", "VIDEO_021", 
    "VIDEO_022", "VIDEO_023", "VIDEO_024"
]

# NOTA: VIDEO_019 no tiene JSON de etiquetas pero lo procesamos igual por si acaso
# NOTA: VIDEO_023 tiene extensión .mp3 pero es realmente un video


def setup_directories():
    """Crear directorios si no existen"""
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)
    
    if not input_path.exists():
        input_path.mkdir(parents=True)
        print(f"⚠️  Carpeta '{INPUT_PATH}' creada.")
        print(f"   Por favor, agrega los videos del otro grupo ahí.")
        return False
    
    if not output_path.exists():
        output_path.mkdir(parents=True)
        print(f"✅ Carpeta '{OUTPUT_PATH}' creada.")
    
    return True


def find_video_file(video_name, input_path):
    """
    Busca el archivo de video correspondiente al nombre.
    Soporta extensiones: .mp4, .mov, .avi, .mp3 (que pueden ser videos mal nombrados)
    """
    input_dir = Path(input_path)
    
    # Buscar con diferentes extensiones (incluyendo .mp3 que puede ser video)
    for ext in ['.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI', '.mp3', '.MP3']:
        # Buscar con el nombre exacto
        video_file = input_dir / f"{video_name}{ext}"
        if video_file.exists():
            return video_file
        
        # Buscar con cualquier prefijo antes del nombre
        for file in input_dir.glob(f"*{video_name}{ext}"):
            return file
    
    return None


def extract_landmarks_from_video(video_path, output_csv_path, pose):
    """
    Extrae landmarks de un video usando MediaPipe Pose.
    
    Args:
        video_path: Path al video
        output_csv_path: Path donde guardar el CSV
        pose: Instancia de MediaPipe Pose
    
    Returns:
        tuple: (success, num_frames, message)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return False, 0, "No se pudo abrir el video"
    
    # Obtener información del video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    landmarks_data = []
    frame_idx = 0
    frames_with_pose = 0
    
    # Procesar cada frame
    pbar = tqdm(total=total_frames, desc=f"  Extrayendo", unit="frames", leave=False)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        # Convertir BGR a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesar con MediaPipe
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Crear diccionario con los datos del frame
            frame_data = {'frame': frame_idx}
            
            # Agregar todas las coordenadas de los landmarks
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                frame_data[f'x_{i}'] = landmark.x
                frame_data[f'y_{i}'] = landmark.y
                frame_data[f'z_{i}'] = landmark.z
                frame_data[f'v_{i}'] = landmark.visibility
            
            landmarks_data.append(frame_data)
            frames_with_pose += 1
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Verificar si se detectaron poses
    if not landmarks_data:
        return False, frame_idx, "No se detectaron poses en el video"
    
    # Guardar a CSV
    df = pd.DataFrame(landmarks_data)
    df.to_csv(output_csv_path, index=False)
    
    detection_rate = (frames_with_pose / frame_idx * 100) if frame_idx > 0 else 0
    message = f"{frames_with_pose}/{frame_idx} frames ({detection_rate:.1f}% detección)"
    
    return True, frames_with_pose, message


def process_other_group_videos():
    """
    Procesa todos los videos del otro grupo.
    """
    print("="*70)
    print("🎬 EXTRACCIÓN DE LANDMARKS - VIDEOS DEL OTRO GRUPO")
    print("="*70)
    print()
    
    # Verificar directorios
    if not setup_directories():
        print("\n❌ No se puede continuar sin los videos.")
        return
    
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)
    
    # Verificar que hay videos
    video_files = (
        list(input_path.glob('*.mp4')) + 
        list(input_path.glob('*.mov')) + 
        list(input_path.glob('*.avi')) + 
        list(input_path.glob('*.mp3'))  # Incluir .mp3 que pueden ser videos
    )
    if not video_files:
        print(f"❌ No se encontraron videos en '{INPUT_PATH}'")
        print("   Formatos soportados: .mp4, .mov, .avi, .mp3")
        return
    
    print(f"📁 Directorio de entrada: {INPUT_PATH}")
    print(f"📁 Directorio de salida: {OUTPUT_PATH}")
    print(f"🎥 Videos encontrados: {len(video_files)}")
    print(f"📋 Videos esperados: {len(EXPECTED_VIDEOS)}")
    print()
    
    # Inicializar MediaPipe Pose
    print("🔧 Inicializando MediaPipe Pose...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✅ MediaPipe listo\n")
    
    # Estadísticas
    stats = {
        'procesados': 0,
        'saltados': 0,
        'errores': 0,
        'total_frames': 0,
        'no_encontrados': []
    }
    
    # Procesar cada video esperado
    print("🔄 Procesando videos...\n")
    
    for video_name in EXPECTED_VIDEOS:
        csv_path = output_path / f"{video_name}.csv"
        
        # Verificar si ya fue procesado
        if csv_path.exists():
            print(f"⏭️  {video_name}: Ya procesado (saltando)")
            stats['saltados'] += 1
            continue
        
        # Buscar el archivo de video
        video_path = find_video_file(video_name, input_path)
        
        if not video_path:
            print(f"❌ {video_name}: Video no encontrado")
            stats['no_encontrados'].append(video_name)
            stats['errores'] += 1
            continue
        
        print(f"▶️  {video_name}: Procesando '{video_path.name}'")
        
        # Extraer landmarks
        success, num_frames, message = extract_landmarks_from_video(
            video_path, csv_path, pose
        )
        
        if success:
            print(f"✅ {video_name}: Guardado → {csv_path.name} ({message})")
            stats['procesados'] += 1
            stats['total_frames'] += num_frames
        else:
            print(f"❌ {video_name}: Error → {message}")
            stats['errores'] += 1
        
        print()
    
    # Cerrar MediaPipe
    pose.close()
    
    # Mostrar resumen
    print("="*70)
    print("📊 RESUMEN DE EXTRACCIÓN")
    print("="*70)
    print(f"✅ Videos procesados exitosamente: {stats['procesados']}")
    print(f"⏭️  Videos saltados (ya procesados): {stats['saltados']}")
    print(f"❌ Videos con errores: {stats['errores']}")
    print(f"🎞️  Total de frames extraídos: {stats['total_frames']:,}")
    
    if stats['no_encontrados']:
        print(f"\n⚠️  Videos no encontrados ({len(stats['no_encontrados'])}):")
        for video in stats['no_encontrados']:
            print(f"   - {video}")
        print("\n   Verifica que los nombres de los archivos coincidan.")
    
    print("\n" + "="*70)
    
    if stats['procesados'] > 0:
        print("✅ Extracción completada!")
        print(f"\n📁 Archivos CSV guardados en: {OUTPUT_PATH}/")
        print("\n🔄 Siguiente paso:")
        print("   cd ../../Entrega2/notebooks")
        print("   python 01_integrate_labels.py")
    else:
        print("⚠️  No se procesó ningún video nuevo.")
    
    print("="*70)


if __name__ == "__main__":
    process_other_group_videos()
