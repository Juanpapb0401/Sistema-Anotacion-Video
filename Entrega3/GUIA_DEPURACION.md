# 🔧 Guía de Depuración - Problema de Detección

## 🚨 Problema Actual

- **Caminar hacia adelante** → detecta como **caminar hacia atrás**
- **Sentarse** → nunca detecta
- **Ponerse de pie** → nunca detecta

## 🔍 Cambios para Depuración

He realizado los siguientes cambios para diagnosticar:

### 1. **activity_classifier.py** - DEBUG Logging

Añadido logging cada 30 frames para ver:
- Número predicho por el modelo (0-4)
- Clase mapeada por `activity_names`
- Valor en `label_encoder.classes_[]`

Esto verificará si el mapeo número→nombre está correcto.

### 2. **real_time_opencv.py** - Deshabilitado Smoothing

```python
# Antes:
prediction = classifier.predict_with_metadata(features)

# Ahora (SIN smoothing):
activity, confidence, probabilities = classifier.predict(features, use_smoothing=False)
```

**Razón:** El smoothing puede esconder el problema real. Sin smoothing vemos predicciones crudas del modelo.

## 🧪 Cómo Probar

### Paso 1: Ejecutar la Aplicación

```bash
cd Entrega3/real_time
../../venv311/bin/python real_time_opencv.py
```

### Paso 2: Observar la Salida en Consola

Cada 30 frames verás:

```
🔍 DEBUG Prediction #30:
   Número predicho: 1
   Clase mapeada: caminar_hacia_camara
   label_encoder.classes_[1] = caminar_hacia_camara
```

**VERIFICAR:**
- ¿El "Número predicho" coincide con lo que haces?
- ¿La "Clase mapeada" es correcta?
- ¿Coinciden "Clase mapeada" y "label_encoder.classes_[]"?

### Paso 3: Probar Cada Actividad

Haz estas acciones **UNA POR UNA** y observa las predicciones:

#### A) Caminar HACIA la cámara (3-4 pasos)
**Esperado:**
- Número: 1
- Clase: `caminar_hacia_camara`
- Probabilidad `caminar_hacia_camara` > 0.5

**Si ves otra cosa:** Anota qué número y qué clase predice.

#### B) Caminar DE REGRESO (3-4 pasos alejándote)
**Esperado:**
- Número: 0
- Clase: `caminar_de_regreso`
- Probabilidad `caminar_de_regreso` > 0.5

**Si ves otra cosa:** Anota qué número y qué clase predice.

#### C) GIRAR 180° (lentamente, 2-3 segundos)
**Esperado:**
- Número: 2
- Clase: `girar`
- Probabilidad `girar` > 0.5

**Si ves otra cosa:** Anota qué número y qué clase predice.

#### D) SENTARSE (LENTO, 3-4 segundos)
**Esperado:**
- Número: 4
- Clase: `sentarse`
- Probabilidad `sentarse` > 0.3 (al menos)

**Si ves otra cosa:** Anota qué número y qué clase predice. ¿Confunde con qué?

#### E) PONERSE DE PIE (LENTO, 3-4 segundos)
**Esperado:**
- Número: 3
- Clase: `ponerse_de_pie`
- Probabilidad `ponerse_de_pie` > 0.3 (al menos)

**Si ves otra cosa:** Anota qué número y qué clase predice. ¿Confunde con qué?

## 📊 Escenarios Posibles

### Escenario 1: Mapeo Invertido

**Síntoma:**
- Caminar HACIA → predice número 0 → "caminar_de_regreso"
- Caminar DE REGRESO → predice número 1 → "caminar_hacia_camara"

**Causa:** El orden alfabético en `label_encoder.classes_` no coincide con cómo se entrenó.

**Solución:** Necesitamos verificar el orden real en `label_encoder.pkl`.

### Escenario 2: Sentarse/Pararse Confundidos con Girar

**Síntoma:**
- Sentarse → predice número 2 → "girar" (probabilidad alta)
- Pararse → predice número 2 → "girar" (probabilidad alta)
- Las probabilidades de `sentarse` y `ponerse_de_pie` son siempre bajas (<0.2)

**Causa:** 
1. El modelo realmente confunde estas acciones (como vimos en evaluation_report.txt)
2. Features de velocidad/aceleración no se calculan correctamente para acciones lentas
3. Necesitas hacer las acciones MÁS LENTO

**Solución:** 
- Re-entrenar con más énfasis en sentarse/pararse
- Agregar features de cambio de altura
- Hacer las acciones MUCHO más lento (3-5 segundos)

### Escenario 3: Features Incorrectas

**Síntoma:**
- Predicciones completamente aleatorias
- Probabilidades muy bajas para todas las clases (<0.3)
- Cambia constantemente entre clases

**Causa:** Las features extraídas en tiempo real no coinciden con las del entrenamiento.

**Solución:** Ejecutar `deep_diagnosis.py` para comparar features.

### Escenario 4: Modelo Siempre Predice Lo Mismo

**Síntoma:**
- No importa qué hagas, siempre predice la misma clase
- Una probabilidad siempre > 0.9
- El número predicho nunca cambia

**Causa:** 
- Scaler no está aplicado correctamente
- Features todas en 0 o valores constantes

**Solución:** Verificar que `video_processor` genera features válidas.

## 🎯 Orden Alfabético Esperado

Según LabelEncoder de sklearn, el orden debería ser:

```
0 → caminar_de_regreso
1 → caminar_hacia_camara
2 → girar
3 → ponerse_de_pie
4 → sentarse
```

**IMPORTANTE:** Este orden se basa en ordenar alfabéticamente:
- caminar_de_regreso
- caminar_hacia_camara
- girar
- ponerse_de_pie
- sentarse

## 📝 Qué Reportar

Después de probar, por favor anota:

1. **Caminar hacia cámara:**
   - Número predicho: ___
   - Clase mostrada: ___
   - Probabilidades: ___

2. **Caminar de regreso:**
   - Número predicho: ___
   - Clase mostrada: ___
   - Probabilidades: ___

3. **Girar:**
   - Número predicho: ___
   - Clase mostrada: ___
   - Probabilidades: ___

4. **Sentarse:**
   - Número predicho: ___
   - Clase mostrada: ___
   - Probabilidades: ___
   - ¿Con qué confunde? ___

5. **Ponerse de pie:**
   - Número predicho: ___
   - Clase mostrada: ___
   - Probabilidades: ___
   - ¿Con qué confunde? ___

6. **Mensaje de DEBUG (cada 30 frames):**
   ```
   Copia aquí el mensaje que aparece en consola
   ```

## 🔧 Próximos Pasos Según Resultado

Una vez tengamos esta información, podremos:

1. **Si es mapeo invertido:** Corregir el mapeo número→nombre
2. **Si confunde sentarse/pararse:** Ajustar modelo o features
3. **Si features incorrectas:** Corregir extracción de features
4. **Si modelo no funciona:** Verificar que modelo/scaler sean correctos

---

**Ejecuta la app y observa los DEBUG messages. La salida en consola es la clave para encontrar el problema.**
