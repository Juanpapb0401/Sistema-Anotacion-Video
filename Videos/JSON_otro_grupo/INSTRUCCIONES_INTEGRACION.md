#  Instrucciones para Integrar Videos del Otro Grupo

##  Objetivo
Integrar los videos etiquetados del otro grupo al dataset de entrenamiento para mejorar el rendimiento del modelo.

##  Pasos Completados
1.  **JSON separado por video**: Modificado el script `split_json_by_video.py`
2.  **Configuración actualizada**: Añadidas las variantes de etiquetas del otro grupo en `config.py`
3.  **Script de integración actualizado**: `01_integrate_labels.py` ahora puede procesar videos del otro grupo

---

##  Pasos Pendientes

### Paso 1: Separar el JSON Grande
```bash
cd "/Users/juanpabloparra/SeptimoSemestre/APO III/ProyectoFinal/Sistema-Anotacion-Video/Videos/JSON_otro_grupo"
python split_json_by_video.py
```

**Resultado esperado:**
- Se creará la carpeta `individual_videos/` con 22 archivos JSON (uno por cada video)
- Archivos: `VIDEO_01.json`, `VIDEO_03.json`, ..., `VIDEO_024.json`

---

### Paso 2: Extraer Features de los Videos del Otro Grupo

**️ IMPORTANTE**: Los videos del otro grupo están en:
```
Entrega1/data/raw_videos_otro_grupo/
```

Necesitas extraer las features de pose usando MediaPipe. Tienes dos opciones:

#### Opción A: Modificar el script existente
Modifica `Entrega1/notebooks/01_extract_landmarks.py` para procesar también los videos del otro grupo.

#### Opción B: Crear un script nuevo
Crea un script específico para extraer features de estos videos:

```python
# Ejemplo: Entrega1/notebooks/01_extract_landmarks_otro_grupo.py
import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path

# Tu código de extracción de landmarks aquí
# Los CSV resultantes deben guardarse en: Entrega1/data/03_features/
# Con nombres: VIDEO_01.csv, VIDEO_03.csv, etc.
```

**Archivos esperados en `Entrega1/data/03_features/`:**
- `VIDEO_01.csv`
- `VIDEO_03.csv`
- `VIDEO_04.csv`
- ... (hasta VIDEO_024.csv)

**Formato del CSV:**
Mismo formato que tus features actuales (frame, nose_x, nose_y, ..., todas las coordenadas de MediaPipe)

---

### Paso 3: Ejecutar la Integración de Etiquetas

Una vez que tengas:
1.  Los JSON individuales en `Videos/JSON_otro_grupo/individual_videos/`
2.  Los CSV de features en `Entrega1/data/03_features/`

Ejecuta:

```bash
cd "/Users/juanpabloparra/SeptimoSemestre/APO III/ProyectoFinal/Sistema-Anotacion-Video/Entrega2/notebooks"
python 01_integrate_labels.py
```

---

##  Resultado Esperado

Después de ejecutar `01_integrate_labels.py`, deberías ver:

```
 Iniciando integración de etiquetas

 Procesando videos y etiquetas...

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

============================================================
📦 INTENTANDO INTEGRAR VIDEOS DEL OTRO GRUPO
============================================================

 Procesando videos del otro grupo...

📹 Procesando: OtroGrupo
  Videos: 100%|████████████| 22/22

 Videos del otro grupo procesados: 22
   Frames adicionales: ~35,000

 Dataset completo COMBINADO guardado: ../data/labeled_dataset_complete.csv
 Dataset principal COMBINADO guardado: ../data/labeled_dataset_main.csv

============================================================
 ESTADÍSTICAS DE INTEGRACIÓN
============================================================

📹 Videos procesados: 64 (42 originales + 22 del otro grupo)
   Por persona:
   - Joshua: 12 videos
   - Juan: 12 videos
   - Santiago: 11 videos
   - Thomas: 12 videos
   - OtroGrupo: 22 videos

🎞️  Frames totales: ~100,000-120,000
   Frames etiquetados: ~90%+
```

---

##  Verificación

### Verificar que los datos se integraron correctamente:

```python
import pandas as pd

# Cargar dataset
df = pd.read_csv("Entrega2/data/labeled_dataset_main.csv")

# Verificar que hay datos del otro grupo
print(df['person'].value_counts())
# Debería aparecer: OtroGrupo    XXXXX

# Verificar distribución de etiquetas
print(df['label'].value_counts())
```

---

## ️ Problemas Comunes

### Problema 1: "No se encontró features CSV"
**Solución**: Ejecuta el paso 2 primero (extraer features con MediaPipe)

### Problema 2: "No se encontró JSON"
**Solución**: Ejecuta `split_json_by_video.py` primero

### Problema 3: Etiquetas no reconocidas
**Solución**: Verifica que el mapeo en `config.py` incluye todas las variantes:
- "Caminar alejandose (espaldas)" → caminar_de_regreso
- "Giro 180 izquierda" → girar
- "Giro 180 derecha" → girar
- "Sentadillas" → sentadilla
- etc.

---

##  Beneficios Esperados

Con 22 videos adicionales (~35,000 frames más):

 **Más datos de entrenamiento** → mejor generalización del modelo
 **Variabilidad adicional** → modelo más robusto
 **Mejor balance de clases** → predicciones más equilibradas
 **Personas diferentes** → menos overfitting a individuos específicos

---

##  Notas Adicionales

- El script detecta automáticamente si hay videos del otro grupo
- Si faltan features, solo procesará los videos originales (sin error)
- Los datasets se guardan como "COMBINADO" cuando incluyen ambos grupos
- Las estadísticas JSON incluyen un campo `includes_other_group: true`

---

## 🆘 ¿Necesitas Ayuda?

Si tienes problemas en algún paso, verifica:
1. Que los paths en `config.py` son correctos
2. Que los nombres de archivos coinciden exactamente
3. Que el formato de los CSV es consistente
