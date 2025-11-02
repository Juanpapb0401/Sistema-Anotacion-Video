# Entrega 2: Preparación de Datos y Entrenamiento de Modelos

## 📋 Descripción

Segunda entrega del proyecto de Sistema de Anotación de Video. En esta fase se integran las etiquetas manuales de Label Studio con los datos de features extraídos, se realiza análisis exploratorio y se entrenan modelos de clasificación.

## 🗂️ Estructura de Archivos

```
Entrega2/
├── data/
│   ├── labeled_dataset_complete.csv       # Dataset con todas las etiquetas
│   ├── labeled_dataset_main.csv           # Dataset solo con las 5 actividades principales
│   ├── train.csv                          # Conjunto de entrenamiento
│   ├── validation.csv                     # Conjunto de validación
│   ├── test.csv                           # Conjunto de prueba
│   ├── label_mapping.json                 # Mapeo de etiquetas
│   ├── integration_statistics.json        # Estadísticas de integración
│   └── EDA_report.txt                     # Reporte de EDA
├── notebooks/
│   ├── config.py                          # Configuración general
│   ├── 01_integrate_labels.py             # Integración de etiquetas
│   ├── 02_eda_labeled.py                  # Análisis exploratorio
│   ├── 03_data_preparation.py             # Preparación para ML
│   ├── 04_model_training.py               # Entrenamiento de modelos
│   └── 05_evaluation.py                   # Evaluación de modelos
├── models/
│   ├── svm_model.pkl                      # Modelo SVM entrenado
│   ├── random_forest_model.pkl            # Modelo Random Forest
│   ├── xgboost_model.pkl                  # Modelo XGBoost
│   └── best_model.pkl                     # Mejor modelo
├── reports/
│   ├── figures/                           # Gráficos del EDA y evaluación
│   └── Entrega2_Informe.pdf              # Documento final
└── README.md                              # Este archivo
```

## 🚀 Instrucciones de Uso

### 1. Instalación de Dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn tqdm
```

### 2. Ejecutar Pipeline Completo

#### Paso 1: Integrar Etiquetas
```bash
cd notebooks
python 01_integrate_labels.py
```

Este script:
- Lee los archivos JSON de Label Studio por cada video
- Mapea las etiquetas a las 5 actividades principales
- Asigna etiquetas a cada frame de los archivos de features
- Genera: `labeled_dataset_complete.csv` y `labeled_dataset_main.csv`

#### Paso 2: Análisis Exploratorio de Datos
```bash
python 02_eda_labeled.py
```

Este script:
- Analiza la distribución de clases
- Compara características por actividad
- Genera visualizaciones y reporte estadístico
- Identifica desbalance de clases y problemas potenciales

#### Paso 3: Preparación de Datos
```bash
python 03_data_preparation.py
```

Este script:
- Divide el dataset en train/validation/test (70/15/15)
- Normaliza características
- Balancea clases usando SMOTE
- Genera features adicionales (velocidades, aceleraciones)

#### Paso 4: Entrenamiento de Modelos
```bash
python 04_model_training.py
```

Este script:
- Entrena múltiples modelos (SVM, Random Forest, XGBoost)
- Realiza ajuste de hiperparámetros con GridSearchCV
- Guarda los modelos entrenados

#### Paso 5: Evaluación de Modelos
```bash
python 05_evaluation.py
```

Este script:
- Evalúa todos los modelos en el conjunto de test
- Genera matrices de confusión
- Calcula métricas (Precision, Recall, F1-Score)
- Identifica el mejor modelo

## 📊 Actividades Clasificadas

El sistema clasifica 5 actividades principales:

1. **caminar_hacia_camara**: Persona caminando acercándose a la cámara
2. **caminar_de_regreso**: Persona caminando alejándose (de espaldas)
3. **girar**: Giros de 180° o 360°
4. **sentarse**: Acción de sentarse en una silla
5. **ponerse_de_pie**: Acción de levantarse de una silla

## 🎯 Métricas de Evaluación

- **Accuracy**: Precisión general del modelo
- **Precision por clase**: Qué tan preciso es el modelo para cada actividad
- **Recall por clase**: Qué tan bien detecta cada actividad
- **F1-Score**: Balance entre Precision y Recall
- **Matriz de Confusión**: Errores entre clases

## 📈 Resultados Esperados

- Dataset balanceado con ~10,000-15,000 frames etiquetados
- Accuracy > 85% en conjunto de test
- F1-Score > 0.80 para todas las clases
- Identificación de actividades más difíciles de clasificar

## 🔧 Configuración

Editar `config.py` para ajustar:
- Rutas de datos
- Mapeo de etiquetas
- Personas y videos incluidos
- Etiquetas a excluir

## 📝 Notas

- Los frames sin etiqueta se excluyen del dataset
- Se descartan actividades adicionales (sentadilla, inclinaciones, sin movimiento)
- Videos lentos se marcan con `video_speed='lento'` para análisis diferenciado
- Se recomienda balancear las clases antes del entrenamiento

## 🐛 Troubleshooting

**Problema**: Archivos JSON no encontrados
- Verificar que los archivos JSON estén en `../Videos/JSON/{persona}/`

**Problema**: Archivos de features no encontrados
- Verificar ruta `../Entrega1/data/03_features/`
- Ejecutar primero los scripts de Entrega1

**Problema**: Dataset desbalanceado
- El script de preparación aplicará SMOTE automáticamente
- Alternativamente, ajustar class_weights en los modelos

## 📧 Contacto

Para dudas o problemas, contactar al equipo de desarrollo.
