# Scripts de Evaluación - Entrega 3


### Lo que ya se hizo en la Entrega 2:
- **Evaluación de modelos** (SVM, Random Forest, XGBoost)
- **Métricas de clasificación** (accuracy, precision, recall, F1-score)
- **Matrices de confusión** para cada modelo
- **Comparación entre modelos** y selección del mejor
- **Análisis por clase** de actividades

 Ubicación: `Entrega2/notebooks/05_evaluation.py` + `Entrega2/data/evaluation_results.json`

### Lo que falta para Entrega 3:

La Entrega 3 requiere análisis **DIFERENTES** que complementan la evaluación de Entrega 2:

1. **Análisis de Reducción de Features** (147 → 15)
   - ¿Cuáles features se seleccionaron y por qué?
   - ¿Cuánta información concentran?
   - ¿Qué tipos de features son?
   - ¿Cuál es el impacto en memoria y velocidad?

2. **Performance en Tiempo Real**
   - ¿Cuántos FPS logra el sistema?
   - ¿Cuál es la latencia por frame?
   - ¿Cuánta CPU y memoria consume?
   - ¿Qué tan estables son las predicciones?

Estos análisis son **NUEVOS** y específicos para el despliegue en tiempo real.

---

##  Cómo Ejecutar

### Opción 1: Ejecutar todo automáticamente

```bash
cd Entrega3
python run_all_evaluations.py
```

Este script:
-  Ejecuta el análisis de features (automático)
- ⚠️ Pregunta si quieres ejecutar el test de performance (requiere webcam)
-  Genera todos los gráficos y reportes
- 📝 Muestra un resumen al final

### Opción 2: Ejecutar scripts individuales

#### 1 Análisis de Reducción de Features (NO requiere webcam)

```bash
cd Entrega3/evaluation
python 01_analyze_feature_reduction.py
```

**Tiempo:** ~30 segundos  
**Genera:**
- 3 gráficos de análisis de features
- Reporte de reducción (.txt)
- Datos en JSON

## Instalación de Dependencias

Necesitas instalar `psutil` para el monitoreo de recursos:

```bash
# Activar ambiente virtual
source venv311/bin/activate

# Instalar psutil
pip install psutil

# O reinstalar todo
pip install -r requirements.txt
```

---

##  Outputs Generados

### Gráficos

1. **01_feature_importance_analysis.png**
   - Importancia de las 15 features seleccionadas (barras horizontales)
   - Importancia acumulada (línea)
   - Indica cuántas features se necesitan para 80% de importancia

2. **02_feature_types_distribution.png**
   - Gráfico de pie con distribución por tipo
   - Gráfico de barras con conteo detallado

3. **03_dimensionality_impact.png**
   - Tabla comparativa: 147 vs 15 features
   - Impacto en memoria, velocidad, complejidad
