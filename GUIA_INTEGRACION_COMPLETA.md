# 🎬 Guía Completa: Integración de Videos del Otro Grupo

## 📋 Resumen del Proceso

```
Videos del otro grupo → Extraer Landmarks → Separar JSON → Integrar todo → Entrenar modelo mejorado
```

---

## ✅ Estado Actual

### ¿Qué tienes?
- ✅ **Videos del otro grupo**: 23 videos en `Entrega1/data/raw_videos_otro_grupo/`
- ✅ **Etiquetas manuales**: JSON grande en `Videos/JSON_otro_grupo/VIDEOS FINAL TALLER LABELING.json`
- ✅ **Scripts listos**: Todos los scripts actualizados y preparados

### ¿Qué falta?
- ⏳ **Landmarks (features)**: Extraer coordenadas de pose de los 23 videos
- ⏳ **JSON separados**: Dividir el JSON grande en archivos individuales
- ⏳ **Integración**: Combinar todo con tu dataset actual

---

## 🚀 PASO A PASO

### **PASO 1: Extraer Landmarks de los Videos** 🎯

#### 1.1. Ir a la carpeta correcta
```bash
cd "/Users/juanpabloparra/SeptimoSemestre/APO III/ProyectoFinal/Sistema-Anotacion-Video/Entrega1/notebooks"
```

#### 1.2. Ejecutar el script de extracción
```bash
python 01_extract_landmarks_otro_grupo.py
```

#### 1.3. ¿Qué va a pasar?
El script va a:
- ✅ Buscar los 23 videos en `raw_videos_otro_grupo/`
- ✅ Extraer las coordenadas de pose con MediaPipe para cada video
- ✅ Crear archivos CSV en `03_features/` con nombres: `VIDEO_01.csv`, `VIDEO_03.csv`, etc.
- ✅ Mostrar progreso en tiempo real para cada video

**Tiempo estimado**: 10-20 minutos (dependiendo de tu computadora)

#### 1.4. Salida esperada:
```
======================================================================
🎬 EXTRACCIÓN DE LANDMARKS - VIDEOS DEL OTRO GRUPO
======================================================================

📁 Directorio de entrada: ../data/raw_videos_otro_grupo
📁 Directorio de salida: ../data/03_features
🎥 Videos encontrados: 23
📋 Videos esperados: 23

🔧 Inicializando MediaPipe Pose...
✅ MediaPipe listo

🔄 Procesando videos...

▶️  VIDEO_01: Procesando 'VIDEO_01.mp4'
  Extrayendo: 100%|████████████| 1479/1479 [00:45<00:00, 32.43frames/s]
✅ VIDEO_01: Guardado → VIDEO_01.csv (1470/1479 frames (99.4% detección))

▶️  VIDEO_03: Procesando 'VIDEO_03.mp4'
  Extrayendo: 100%|████████████| 340/340 [00:10<00:00, 33.12frames/s]
✅ VIDEO_03: Guardado → VIDEO_03.csv (335/340 frames (98.5% detección))

... (continúa para todos los videos)

======================================================================
📊 RESUMEN DE EXTRACCIÓN
======================================================================
✅ Videos procesados exitosamente: 23
⏭️  Videos saltados (ya procesados): 0
❌ Videos con errores: 0
🎞️  Total de frames extraídos: 35,478

======================================================================
✅ Extracción completada!

📁 Archivos CSV guardados en: ../data/03_features/

🔄 Siguiente paso:
   cd ../../Entrega2/notebooks
   python 01_integrate_labels.py
======================================================================
```

---

### **PASO 2: Separar el JSON Grande** 📄

#### 2.1. Ir a la carpeta de JSON
```bash
cd "/Users/juanpabloparra/SeptimoSemestre/APO III/ProyectoFinal/Sistema-Anotacion-Video/Videos/JSON_otro_grupo"
```

#### 2.2. Ejecutar el separador
```bash
python split_json_by_video.py
```

#### 2.3. Salida esperada:
```
============================================================
🎬 SEPARADOR DE JSON POR VIDEO
============================================================

📂 Leyendo archivo: VIDEOS FINAL TALLER LABELING.json
✅ Total de tareas en el archivo: 23

🔗 Consolidando datos...

💾 Guardado: individual_videos/VIDEO_01.json (1 tarea(s))
💾 Guardado: individual_videos/VIDEO_03.json (1 tarea(s))
💾 Guardado: individual_videos/VIDEO_04.json (1 tarea(s))
... (continúa)

✅ Proceso completado!
   - 23 archivos JSON creados
   - Ubicación: individual_videos

============================================================
🎉 23 videos procesados exitosamente!
============================================================
```

---

### **PASO 3: Integrar Todo** 🔗

#### 3.1. Ir a notebooks de Entrega2
```bash
cd "/Users/juanpabloparra/SeptimoSemestre/APO III/ProyectoFinal/Sistema-Anotacion-Video/Entrega2/notebooks"
```

#### 3.2. Ejecutar integración
```bash
python 01_integrate_labels.py
```

#### 3.3. Salida esperada:
```
🚀 Iniciando integración de etiquetas

🔄 Procesando videos y etiquetas...

📹 Procesando: Joshua
  Videos normales: 100%|████████████| 10/10
  Videos lentos: 100%|████████████| 2/2

📹 Procesando: Juan
  Videos normales: 100%|████████████| 10/10
  Videos lentos: 100%|████████████| 2/2

📹 Procesando: Santiago
  Videos normales: 100%|████████████| 11/11

📹 Procesando: Thomas
  Videos normales: 100%|████████████| 10/10
  Videos lentos: 100%|████████████| 2/2

🔗 Consolidando datos...

✅ Dataset completo guardado: ../data/labeled_dataset_complete.csv
✅ Dataset principal guardado: ../data/labeled_dataset_main.csv

============================================================
📦 INTENTANDO INTEGRAR VIDEOS DEL OTRO GRUPO
============================================================

🔄 Procesando videos del otro grupo...

📹 Procesando: OtroGrupo
  Videos: 100%|████████████| 22/22

✅ Videos del otro grupo procesados: 22
   Frames adicionales: 35,478

✅ Dataset completo COMBINADO guardado: ../data/labeled_dataset_complete.csv
✅ Dataset principal COMBINADO guardado: ../data/labeled_dataset_main.csv
✅ Estadísticas actualizadas: ../data/integration_statistics.json

============================================================
📊 ESTADÍSTICAS DE INTEGRACIÓN
============================================================

📹 Videos procesados: 64
   Por persona:
   - Joshua: 12 videos
   - Juan: 12 videos
   - Santiago: 11 videos
   - Thomas: 12 videos
   - OtroGrupo: 22 videos

🎞️  Frames totales: 105,847
   Frames etiquetados: 98,234 (92.8%)

🎯 Frames con etiquetas principales: 87,456
   Frames excluidos: 10,778

🏷️  Distribución por etiqueta:
   ✓ caminar_hacia_camara: 24,563 (25.0%)
   ✓ caminar_de_regreso: 23,891 (24.3%)
   ✓ girar: 18,234 (18.6%)
   ✓ sentarse: 10,789 (11.0%)
   ✓ ponerse_de_pie: 9,979 (10.2%)
   ✗ sin_movimiento: 7,234 (7.4%)
   ✗ sentadilla: 2,345 (2.4%)
   ✗ inclinacion_lateral: 1,199 (1.2%)

============================================================

✅ Integración completada exitosamente!

📁 Archivos generados en: ../data/
   - labeled_dataset_complete.csv (todos los datos)
   - labeled_dataset_main.csv (solo etiquetas principales)
   - integration_statistics.json
   - label_mapping.json
```

---

## 📊 Verificación de Resultados

### Verificar que todo se integró correctamente:

```python
import pandas as pd

# Cargar el dataset
df = pd.read_csv("Entrega2/data/labeled_dataset_main.csv")

# Ver resumen
print(f"Total de frames: {len(df):,}")
print(f"\nFrames por persona:")
print(df['person'].value_counts())
print(f"\nFrames por etiqueta:")
print(df['label'].value_counts())

# Verificar que hay datos del otro grupo
otro_grupo = df[df['person'] == 'OtroGrupo']
print(f"\n✅ Frames del otro grupo: {len(otro_grupo):,}")
```

**Salida esperada:**
```
Total de frames: 87,456

Frames por persona:
Joshua        15,234
OtroGrupo     22,189
Thomas        14,567
Juan          13,891
Santiago      21,575

Frames por etiqueta:
caminar_hacia_camara    24,563
caminar_de_regreso      23,891
girar                   18,234
sentarse                10,789
ponerse_de_pie           9,979

✅ Frames del otro grupo: 22,189
```

---

## 🎓 PASO 4: Re-entrenar el Modelo

Una vez que tengas el dataset combinado, re-ejecuta el entrenamiento:

```bash
cd "/Users/juanpabloparra/SeptimoSemestre/APO III/ProyectoFinal/Sistema-Anotacion-Video/Entrega2/notebooks"

# Opcional: regenerar el EDA
python 02_eda_labeled.py

# Re-preparar datos (con más samples)
python 03_data_preparation.py

# Re-entrenar modelos (con más datos = mejor modelo)
python 04_model_training.py

# Evaluar
python 05_evaluation.py
```

---

## 💡 Mejoras Esperadas

Con ~35,000 frames adicionales:

📈 **Mejora en métricas**:
- Accuracy: +3-5% esperado
- F1-Score: +2-4% esperado
- Mejor generalización a personas nuevas

🎯 **Mejor balance de clases**:
- Más ejemplos de actividades minoritarias
- Distribución más equilibrada

🚀 **Modelo más robusto**:
- Menos overfitting
- Mejor rendimiento en datos reales
- Mayor confianza en predicciones

---

## ⚠️ Notas Importantes

1. **VIDEO_019**: No tiene JSON de etiquetas, pero extraeremos sus features por si acaso
2. **VIDEO_023.mp3**: Es realmente un video (extensión incorrecta), el script lo maneja automáticamente
3. **Tiempo total**: ~30-40 minutos para todo el proceso
4. **Espacio en disco**: ~500MB adicionales para los CSV de features

---

## 🆘 Solución de Problemas

### Problema: "No se detectaron poses en el video"
**Solución**: Algunos frames pueden no tener personas visibles. Es normal si la detección es >95%.

### Problema: "No se encontró features CSV"
**Solución**: Verifica que el PASO 1 se completó correctamente. Los CSV deben estar en `Entrega1/data/03_features/`

### Problema: "Out of memory"
**Solución**: Procesa los videos en lotes más pequeños o reduce la resolución.

---

## ✅ Lista de Verificación

Antes de continuar al siguiente paso, verifica:

- [ ] PASO 1: ¿Se crearon 23 archivos CSV en `03_features/`?
- [ ] PASO 2: ¿Se crearon 22 archivos JSON en `individual_videos/`?
- [ ] PASO 3: ¿El archivo `labeled_dataset_main.csv` incluye "OtroGrupo" en la columna `person`?
- [ ] PASO 4: ¿Las métricas del modelo mejoraron después de re-entrenar?

---

## 🎯 Comando Rápido (Todo en Uno)

Si quieres ejecutar todo de una vez:

```bash
# Desde la raíz del proyecto
cd "Entrega1/notebooks" && python 01_extract_landmarks_otro_grupo.py && \
cd "../../Videos/JSON_otro_grupo" && python split_json_by_video.py && \
cd "../../Entrega2/notebooks" && python 01_integrate_labels.py && \
python 03_data_preparation.py && python 04_model_training.py
```

---

¡Listo! Ahora tienes todo preparado para integrar los videos del otro grupo y mejorar tu modelo 🚀
