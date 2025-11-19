#  Sistema de Clasificación de Actividades Humanas en Tiempo Real

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange.svg)](https://mediapipe.dev/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1-red.svg)](https://xgboost.readthedocs.io/)

> Sistema de reconocimiento de actividades humanas (HAR) utilizando visión por computadora y machine learning para clasificación en tiempo real.

**Universidad ICESI** | Algoritmos y Programación III  
**Proyecto Final** | Entrega 3 - Despliegue y Evaluación  
**Fecha**: Noviembre 2025

---

##  Video de Presentación

[![Ver Video en YouTube](https://img.shields.io/badge/▶️_Ver_Video-YouTube-red?style=for-the-badge&logo=youtube)](https://youtu.be/wfaI14Zmnc8)

> **Duración**: ~9 minutos | **Contenido**: Demostración completa del sistema, técnicas utilizadas, resultados y análisis de performance

---

##  Tener en cuenta

Se utilizaron videos del grupo de Luis Manuel Rojas de la Universidad Icesi.
La carpeta donde estan ubicados los Landmarks tanto de los videos de nuestro grupo como el del otro grupo, no se subieron al readme al igual que los archivos de los modelos realizados, por razones de seguridad y porque pesan mucho.

##  Descripción General

Sistema inteligente de análisis de video que automatiza el reconocimiento de actividades humanas (HAR) mediante:

- **Detección de pose en tiempo real** con MediaPipe (33 landmarks corporales)
- **Clasificación de 5 actividades** usando XGBoost optimizado
- **Reducción dimensional inteligente** de 147 → 15 características clave
- **Interfaz visual en vivo** con OpenCV mostrando predicciones y confianza

###  Actividades Reconocidas

1.  **Caminar hacia la cámara**
2.  **Caminar de regreso** (alejándose)
3.  **Girar** (rotación corporal)
4.  **Sentarse**
5.  **Ponerse de pie**

---

##  Características Principales

###  Sistema en Tiempo Real
- **Alto rendimiento**: 25-30 FPS en hardware estándar
- **Baja latencia**: ~35-40 ms por frame
- **Eficiencia computacional**: Uso optimizado de CPU/RAM
- **Estabilidad**: Predicciones suavizadas con smoothing temporal

###  Modelo Inteligente
- **Accuracy**: 77.91% (validación) | 64.29% (test)
- **F1-Score**: 0.7699 (validación) | 0.6494 (test)
- **Algoritmo**: XGBoost con hiperparámetros optimizados
- **Features**: 15 características seleccionadas de 147 originales (89.8% reducción)

### Análisis Avanzado
- Extracción de **33 landmarks corporales** (MediaPipe Pose)
- Cálculo de **ángulos articulares** (codos, rodillas, caderas)
- Estimación de **velocidades y aceleraciones**
- **Normalización espacial** para invariancia a posición/escala

###  Interfaz Visual
- Visualización en tiempo real del esqueleto 2D
- Etiquetas de actividad con nivel de confianza
- Métricas de performance (FPS) en pantalla
- Códigos de color para diferentes partes del cuerpo

---

## 🏆 Resultados Destacados

### Performance del Modelo

| Métrica | Validación | Test |
|---------|------------|------|
| **Accuracy** | 77.91% | 64.29% |
| **Precision** | 77.83% | 66.42% |
| **Recall** | 77.91% | 64.29% |
| **F1-Score** | 76.99% | 64.94% |


### Reducción de Características

- **Dimensionalidad**: 147 → 15 features (**89.8% reducción**)
- **Top 5 features**: Ángulos de rodillas y codos concentran ~46% de importancia
- **Beneficios**: 
  - 80-85% más rápido en inferencia
  - 90% menos uso de memoria
  - Modelo más interpretable

---

##  Arquitectura del Sistema

```
┌─────────────────┐
│   Video Input   │ (Webcam / Archivo)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   MediaPipe     │ Extracción de 33 landmarks
│   Pose Engine   │ (x, y, z, visibility)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Feature Engine  │ 147 features → Normalización
│                 │ Ángulos + Distancias + Velocidades
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ StandardScaler  │ Escalado (media=0, std=1)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Feature Selector │ Selección de 15 features clave
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  XGBoost Model  │ Clasificación (5 clases)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Temporal Filter │ Smoothing de predicciones
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Video Output +  │ Visualización + Etiquetas
│  Predictions    │
└─────────────────┘
```

---

##  Instalación

### Requisitos del Sistema

- **Sistema Operativo**: macOS, Linux, Windows
- **Python**: 3.11 (recomendado) o 3.8-3.11
- **Webcam**: Necesaria para clasificación en vivo
- **RAM**: Mínimo 4GB, recomendado 8GB
- **CPU**: Procesador moderno multi-core

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/Juanpapb0401/Sistema-Anotacion-Video.git
cd Sistema-Anotacion-Video
```

### Paso 2: Crear Ambiente Virtual

```bash
# Crear ambiente virtual con Python 3.11
python3.11 -m venv venv311

# Activar ambiente virtual
# En macOS/Linux:
source venv311/bin/activate

# En Windows:
venv311\Scripts\activate
```

### Paso 3: Instalar Dependencias

```bash
# Instalar todas las dependencias
pip install -r requirements.txt
```


##  Uso Rápido

### Clasificación en Tiempo Real (Webcam)

```bash
cd Entrega3/real_time
python real_time_opencv.py
```

**Controles**:
- `Espacio`: Pausar/Reanudar
- `Q`: Salir de la aplicación

### Ejecutar Pipeline Completo (Desde cero)

#### Entrega 1: Extracción de Landmarks

```bash
cd Entrega1/notebooks
python 01_extract_landmarks.py
python 02_preprocess_landmarks.py
python 03_compute_features.py
```

#### Entrega 2: Entrenamiento del Modelo

```bash
cd Entrega2/notebooks
python run_pipeline.py  # Ejecuta todo el pipeline
```

O ejecutar paso a paso:
```bash
python 01_integrate_labels.py
python 02_eda_labeled.py
python 03_data_preparation.py
python 04_model_training.py
python 05_evaluation.py
```

#### Entrega 3: Evaluación y Despliegue

```bash
cd Entrega3
python run_all_evaluations.py  # Ejecuta análisis de features y performance
```


##  Metodología

El proyecto sigue la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining):

### 1️ Comprensión del Negocio
- Definición del problema de reconocimiento de actividades
- Identificación de 5 actividades objetivo
- Establecimiento de métricas de éxito

### 2 Comprensión de los Datos
- Captura de videos de actividades
- Extracción de landmarks con MediaPipe
- Análisis exploratorio de datos (EDA)

### 3 Preparación de los Datos
- Integración de etiquetas
- Normalización de coordenadas
- Feature engineering (ángulos, velocidades)
- Balanceo con SMOTE
- División en train/validation/test (70/15/15)

### 4 Modelado
- Entrenamiento de múltiples modelos (SVM, Random Forest, XGBoost)
- Optimización de hiperparámetros
- Selección de features (147 → 15)
- Validación cruzada

### 5 Evaluación
- Métricas de clasificación (accuracy, precision, recall, F1)
- Matrices de confusión
- Análisis de errores por clase
- Evaluación de performance en tiempo real

### 6 Despliegue
- Sistema de clasificación en tiempo real
- Interfaz visual con OpenCV
- Optimización de velocidad (FPS)
- Documentación completa

---

##  Tecnologías Utilizadas

### Lenguajes y Frameworks
- **Python 3.11**: Lenguaje principal
- **NumPy 1.26**: Operaciones numéricas
- **Pandas 2.3**: Manipulación de datos

### Computer Vision
- **OpenCV 4.12**: Procesamiento de video y visualización
- **MediaPipe 0.10**: Detección de pose y landmarks

### Machine Learning
- **Scikit-learn 1.7**: Preprocessing, métricas, SVM, Random Forest
- **XGBoost 3.1**: Modelo de clasificación principal
- **Imbalanced-learn 0.14**: Balanceo de clases (SMOTE)

### Visualización y Análisis
- **Matplotlib 3.10**: Gráficos y visualizaciones
- **Seaborn 0.13**: Visualizaciones estadísticas

### Utilidades
- **Joblib 1.5**: Serialización de modelos
- **psutil 7.3**: Monitoreo de recursos del sistema
