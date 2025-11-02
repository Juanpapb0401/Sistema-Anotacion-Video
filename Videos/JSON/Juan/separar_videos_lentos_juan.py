import json
import os

# Configuración de rutas
INPUT_FILE = "1-2Lento.json"
OUTPUT_DIR = "videos_individuales"

def separar_videos_lentos():
    """
    Separa cada video del archivo JSON de videos lentos en archivos JSON individuales
    """
    # Crear carpeta de salida si no existe
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"✓ Carpeta '{OUTPUT_DIR}' creada")
    
    # Leer el archivo JSON
    print(f"\n→ Leyendo archivo: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        videos = json.load(f)
    
    print(f"✓ Se encontraron {len(videos)} videos en el archivo")
    
    # Procesar cada video
    videos_creados = 0
    for video in videos:
        inner_id = video.get('inner_id')
        
        # Crear nombre del archivo de salida
        output_filename = f"Lento{inner_id}Juan.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Guardar el video individual en un nuevo archivo JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([video], f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Creado: {output_filename} (inner_id: {inner_id})")
        videos_creados += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Proceso completado exitosamente")
    print(f"  Total de archivos JSON creados: {videos_creados}")
    print(f"  Ubicación: {OUTPUT_DIR}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    separar_videos_lentos()
