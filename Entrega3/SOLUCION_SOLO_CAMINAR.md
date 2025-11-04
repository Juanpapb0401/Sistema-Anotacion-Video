# 🚨 PROBLEMA CRÍTICO: Solo Predice "Caminar"

## ❌ Síntoma
El modelo **SOLO** predice:
- ✅ caminar_hacia_camara
- ✅ caminar_de_regreso
- ❌ **NUNCA** girar
- ❌ **NUNCA** sentarse  
- ❌ **NUNCA** ponerse_de_pie

## 🔍 Diagnóstico

Este problema indica que:

### **Hipótesis 1: Features Incorrectas (MUY PROBABLE - 90%)**
Las features extraídas de la cámara son **completamente diferentes** a las del entrenamiento. El scaler transforma estas features de tal manera que SIEMPRE se parecen a "caminar".

**Evidencia:**
- El modelo funciona en `evaluation_report.txt` (86% accuracy)
- Pero en tiempo real solo predice 2 de 5 clases
- Las probabilidades de girar/sentarse/pararse son SIEMPRE bajas (<0.1)

### **Hipótesis 2: Modelo Colapsado (POCO PROBABLE - 10%)**
El modelo está roto y solo aprendió a predecir "caminar".

**Evidencia en contra:**
- El `evaluation_report.txt` muestra que SÍ predijo las 5 clases
- F1-scores: 0.73-0.91 en todas las clases

## ✅ Qué Hacer AHORA

### Paso 1: Ejecuta la App con Debug Mejorado

```bash
cd Entrega3/real_time
../../venv311/bin/python real_time_opencv.py
```

**Observa en CONSOLA:**
```
[Frame 30] ✅ caminar_hacia_camara: 65.32% | ✅ caminar_de_regreso: 25.10% | girar: 5.21% | sentarse: 3.15% | ponerse_de_pie: 1.22% | → CAMINAR_HACIA_CAMARA
```

**Observa en PANTALLA:**
- Todas las 5 probabilidades mostradas
- Color verde para la elegida
- Ordenadas de mayor a menor

### Paso 2: Prueba Estas Acciones

1. **PARADO SIN MOVERTE** (5 segundos)
   - ¿Qué predice?
   - ¿Las probabilidades están distribuidas o una domina?

2. **SENTARTE MUY LENTO** (5 segundos, exagerado)
   - Observa las probabilidades MIENTRAS te sientas
   - ¿Sube "sentarse"? ¿Aunque sea a 15-20%?
   - ¿O siempre está en 5%?

3. **GIRAR 180° MUY LENTO** (5 segundos, paso a paso)
   - ¿Sube "girar"? ¿Aunque sea a 20-30%?
   - ¿O siempre está en 5%?

### Paso 3: Anota los Resultados

**Caso A: Probabilidades NUNCA suben**
```
Parado:   caminar_hacia: 60%, caminar_regreso: 30%, girar: 5%, sentarse: 3%, pararse: 2%
Sentando: caminar_hacia: 55%, caminar_regreso: 35%, girar: 6%, sentarse: 3%, pararse: 1%
Girando:  caminar_hacia: 50%, caminar_regreso: 40%, girar: 7%, sentarse: 2%, pararse: 1%
```

**→ Features INCORRECTAS en tiempo real**

**Caso B: Probabilidades SÍ suben pero no ganan**
```
Parado:   caminar_hacia: 45%, caminar_regreso: 30%, girar: 15%, sentarse: 7%, pararse: 3%
Sentando: caminar_hacia: 35%, caminar_regreso: 25%, girar: 15%, sentarse: 20%, pararse: 5%
Girando:  caminar_hacia: 30%, caminar_regreso: 25%, girar: 35%, sentarse: 7%, pararse: 3%
```

**→ Modelo funciona PERO necesita threshold más bajo o re-entrenamiento**

## 🛠️ Soluciones Según Caso

### Solución para Caso A: Features Incorrectas

**OPCIÓN 1: Re-extraer features de videos y re-entrenar** (MEJOR)
```bash
cd Entrega2/notebooks
# Re-ejecutar TODO el pipeline
python 01_integrate_labels.py
python 02_eda_labeled.py
python 03_data_preparation.py
python 04_model_training_fast.py
python 05_evaluation.py
```

**OPCIÓN 2: Verificar y corregir video_processor.py**
- Ejecutar `diagnose_features.py` para ver qué features difieren
- Corregir la extracción de features en `video_processor.py`
- Asegurar que genera exactamente las mismas 147 features

### Solución para Caso B: Modelo Funciona Pero Débil

**OPCIÓN 1: Bajar threshold de confianza**
```python
# real_time_opencv.py, línea ~77
confidence_threshold = 0.25  # Antes: 0.4
```

**OPCIÓN 2: Re-entrenar con más énfasis en otras clases**
```python
# 04_model_training.py
# Agregar class_weight='balanced' o sample_weight
```

**OPCIÓN 3: Hacer acciones MÁS LENTAS Y EXAGERADAS**
- Sentarse: 5-7 segundos (no 1-2)
- Girar: 4-5 segundos (paso a paso)
- Pararse: 5-7 segundos (muy lento)

## 🎯 Qué Esperar

### Si es Caso A (features incorrectas):
- **NO** habrá solución rápida
- Necesitas re-entrenar TODO
- Toma 1-2 horas

### Si es Caso B (modelo débil):
- Bajar threshold puede ayudar ALGO
- Re-entrenar con balance mejorará mucho
- Acciones lentas ayudan temporalmente

## 📊 Evidencia Actual

Según `evaluation_report.txt`, el modelo SÍ aprendió las 5 clases:

```
caminar_de_regreso:   F1=0.859 ✅
caminar_hacia_camara: F1=0.910 ✅
girar:                F1=0.870 ✅
ponerse_de_pie:       F1=0.832 ✅
sentarse:             F1=0.735 ⚠️
```

Pero confunde:
- **sentarse → girar: 63 veces** ❌
- **girar → sentarse: 59 veces** ❌

Esto sugiere que sentarse/girar son MUY similares para el modelo.

## 🚀 PLAN DE ACCIÓN

### INMEDIATO (5 minutos):
1. Ejecutar app con debug mejorado
2. Observar probabilidades en CONSOLA y PANTALLA
3. Anotar qué caso es (A o B)

### CORTO PLAZO (1 hora):
- **Si Caso A:** Re-entrenar desde `03_data_preparation.py`
- **Si Caso B:** Bajar threshold a 0.25 y hacer acciones lentas

### LARGO PLAZO (para Entrega 4):
1. Agregar features de altura (hip_y, shoulder_y)
2. Cambiar a modelo temporal (LSTM)
3. Grabar más videos de sentarse/pararse/girar
4. Usar data augmentation

---

## 📝 TEMPLATE para Reportar

```
RESULTADOS DEL TEST:

1. Parado sin moverte:
   - caminar_hacia: ___%
   - caminar_regreso: ___%
   - girar: ___%
   - sentarse: ___%
   - pararse: ___%

2. Sentándote LENTO (5 seg):
   - caminar_hacia: ___%
   - caminar_regreso: ___%
   - girar: ___%
   - sentarse: ___%
   - pararse: ___%

3. Girando 180° LENTO (5 seg):
   - caminar_hacia: ___%
   - caminar_regreso: ___%
   - girar: ___%
   - sentarse: ___%
   - pararse: ___%

¿Es Caso A o Caso B?: _____
```

**Ejecuta la app y llena este template. Con eso sabremos exactamente qué hacer.**
