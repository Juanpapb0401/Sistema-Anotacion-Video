"""
Script para separar un JSON de Label Studio en archivos individuales por video.
Cada video tendrá su propio archivo JSON con sus anotaciones correspondientes.
"""

import json
import os
import re
from pathlib import Path


def extract_video_name(file_upload):
    """
    Extrae el nombre limpio del video desde el campo file_upload.
    Ejemplo: "adac5c47-VIDEO_01.mp4" -> "VIDEO_01"
    """
    if not file_upload:
        return None
    
    # Buscar el patrón VIDEO_XX
    match = re.search(r'VIDEO_\d+', file_upload, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    
    # Si no encuentra el patrón, usar el nombre completo sin extensión
    filename = os.path.basename(file_upload)
    name_without_ext = os.path.splitext(filename)[0]
    # Remover el hash al inicio si existe
    if '-' in name_without_ext:
        parts = name_without_ext.split('-', 1)
        if len(parts) > 1:
            return parts[1]
    
    return name_without_ext


def split_json_by_video(input_file, output_dir):
    """
    Lee el JSON de Label Studio y crea un archivo JSON por cada video.
    
    Args:
        input_file: Ruta al archivo JSON de entrada
        output_dir: Directorio donde se guardarán los archivos separados
    """
    # Crear directorio de salida si no existe
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Leer el archivo JSON
    print(f" Leyendo archivo: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f" Total de tareas en el archivo: {len(data)}")
    
    # Diccionario para agrupar por video
    videos_dict = {}
    videos_without_name = []
    
    # Agrupar las tareas por video
    for task in data:
        file_upload = task.get('file_upload', '')
        video_name = extract_video_name(file_upload)
        
        if video_name:
            if video_name not in videos_dict:
                videos_dict[video_name] = []
            videos_dict[video_name].append(task)
        else:
            videos_without_name.append(task)
    
    # Guardar cada video en su propio archivo
    saved_count = 0
    for video_name, tasks in videos_dict.items():
        # Crear nombre de archivo limpio
        output_file = output_path / f"{video_name}.json"
        
        # Guardar el JSON (array con una sola tarea)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)
        
        print(f" Guardado: {output_file} ({len(tasks)} tarea(s))")
        saved_count += 1
    
    # Reportar videos sin nombre
    if videos_without_name:
        print(f"\n️  Advertencia: {len(videos_without_name)} tarea(s) sin nombre de video identificable")
        print("   Estas tareas no fueron guardadas en archivos individuales.")
    
    print(f"\n Proceso completado!")
    print(f"   - {saved_count} archivos JSON creados")
    print(f"   - Ubicación: {output_dir}")
    
    return saved_count


def main():
    """Función principal del script."""
    # Rutas
    script_dir = Path(__file__).parent
    input_file = script_dir / "VIDEOS FINAL TALLER LABELING.json"
    output_dir = script_dir / "individual_videos"
    
    print("=" * 60)
    print("🎬 SEPARADOR DE JSON POR VIDEO")
    print("=" * 60)
    print()
    
    if not input_file.exists():
        print(f" Error: No se encontró el archivo {input_file}")
        return
    
    # Ejecutar la separación
    saved_count = split_json_by_video(input_file, output_dir)
    
    print()
    print("=" * 60)
    print(f"🎉 {saved_count} videos procesados exitosamente!")
    print("=" * 60)


if __name__ == "__main__":
    main()
