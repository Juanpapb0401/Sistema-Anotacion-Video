import json
import os

# Ruta al archivo JSON de Juan
input_file = '/Users/juanpabloparra/SeptimoSemestre/APO III/ProyectoFinal/Sistema-Anotacion-Video/Videos/JSON/Juan/1-10Juan.json'

# Carpeta de salida para los archivos individuales
output_folder = '/Users/juanpabloparra/SeptimoSemestre/APO III/ProyectoFinal/Sistema-Anotacion-Video/Videos/JSON/Juan/videos_individuales'

# Crear carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Leer el archivo JSON
print(f"Leyendo archivo: {input_file}")
with open(input_file, 'r', encoding='utf-8') as f:
    videos = json.load(f)

print(f"Total de videos encontrados: {len(videos)}")

# Separar cada video en su propio archivo
for video in videos:
    inner_id = video.get('inner_id')
    
    if inner_id is None:
        print(f"️ Video sin inner_id encontrado, saltando...")
        continue
    
    # Nombre del archivo de salida
    output_filename = f"{inner_id}Juan.json"
    output_path = os.path.join(output_folder, output_filename)
    
    # Guardar el video individual en un archivo JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([video], f, indent=2, ensure_ascii=False)
    
    print(f"✓ Video {inner_id} guardado en: {output_filename}")

print(f"\n{'='*50}")
print(f"✓ Proceso completado!")
print(f"  Total de archivos creados: {len(videos)}")
print(f"  Ubicación: {output_folder}")
print(f"{'='*50}")
