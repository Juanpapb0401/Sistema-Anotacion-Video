# ⚡ Cambios para Depuración - Resumen Rápido

## 🎯 Objetivo
Diagnosticar por qué el modelo confunde las clases (especialmente caminar hacia/atrás y sentarse/pararse).

## ✅ Cambios Realizados

### 1. `activity_classifier.py`
- ✅ Añadido DEBUG logging cada 30 frames
- ✅ Muestra: número predicho, clase mapeada, valor de label_encoder
- ✅ Corregido `class_counts` para usar índices numéricos

### 2. `real_time_opencv.py`  
- ✅ Deshabilitado smoothing temporalmente (`use_smoothing=False`)
- ✅ Esto muestra predicciones CRUDAS del modelo sin suavizado
- ✅ Más fácil ver qué predice realmente frame por frame

### 3. Documentación
- ✅ `GUIA_DEPURACION.md` - Guía detallada de qué probar y qué observar
- ✅ Lista los 5 escenarios posibles de error
- ✅ Explica qué reportar para diagnóstico

## 🚀 Cómo Usar

```bash
cd Entrega3/real_time
../../venv311/bin/python real_time_opencv.py
```

## 👀 Qué Observar

### En Consola (cada 30 frames):
```
🔍 DEBUG Prediction #30:
   Número predicho: 1
   Clase mapeada: caminar_hacia_camara
   label_encoder.classes_[1] = caminar_hacia_camara
```

### En Pantalla:
- Probabilidades de TODAS las clases
- Clase elegida
- Confianza

## 🧪 Probar en Este Orden

1. **Caminar HACIA cámara** → ¿Predice 0 o 1? ¿Qué nombre?
2. **Caminar DE REGRESO** → ¿Predice 0 o 1? ¿Qué nombre?
3. **Girar 180°** → ¿Predice 2? ¿Qué nombre?
4. **Sentarse LENTO** → ¿Predice 4? ¿O confunde con 2 (girar)?
5. **Pararse LENTO** → ¿Predice 3? ¿O confunde con 2 (girar)?

## 📊 Orden Esperado (Alfabético)

```
0 → caminar_de_regreso
1 → caminar_hacia_camara  
2 → girar
3 → ponerse_de_pie
4 → sentarse
```

## 🔍 Hipótesis Principales

### Hipótesis 1: Clases Invertidas
- Caminar hacia → predice 0 (debería ser 1)
- Caminar atrás → predice 1 (debería ser 0)
- **Fix:** Corregir mapeo

### Hipótesis 2: Sentarse/Pararse Confundidos
- Sentarse → predice 2 (girar)
- Pararse → predice 2 (girar)
- **Fix:** Re-entrenar o hacer acciones MÁS LENTO

### Hipótesis 3: Features Incorrectas
- Predicciones aleatorias
- Probabilidades bajas
- **Fix:** Verificar extracción de features

## 📝 Archivos Modificados

```
Entrega3/
├── real_time/
│   ├── activity_classifier.py    [DEBUG logging, class_counts fix]
│   └── real_time_opencv.py       [smoothing=False, direct predict]
├── GUIA_DEPURACION.md           [Guía detallada]
├── RESUMEN_DEBUG.md             [Este archivo]
├── deep_diagnosis.py            [Herramienta completa diagnóstico]
├── test_model_predictions.py    [Test modelo con test.csv]
└── verify_label_encoding.py     [Verificar label_encoder]
```

## 🎬 Siguiente Paso

**EJECUTAR LA APP Y OBSERVAR EL DEBUG OUTPUT EN CONSOLA**

Los mensajes de debug te dirán exactamente qué número predice el modelo y cómo se mapea a nombres. Con eso podremos identificar si:

1. El modelo predice correctamente pero el mapeo está mal
2. El modelo predice mal (confunde clases)
3. Las features están mal

---

**Ver `GUIA_DEPURACION.md` para instrucciones detalladas.**
