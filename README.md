# Sistema de Anotación de Video para Análisis de Actividades Humanas

Este proyecto es una herramienta de software desarrollada para el curso de Algoritmos y Programación III de la Universidad ICESI. El sistema utiliza visión por computadora para clasificar actividades humanas en tiempo real y realizar un seguimiento de los movimientos articulares.

## Descripción del Problema

El objetivo es automatizar el proceso de reconocimiento de actividades humanas (Human Activity Recognition - HAR) a partir de una señal de video. El sistema está diseñado para identificar un conjunto predefinido de acciones y extraer datos posturales mediante el seguimiento de landmarks corporales, abordando la necesidad de un análisis rápido y objetivo sin supervisión humana constante.

## Características Principales

* **Detección de Esqueleto en 2D**: Utiliza MediaPipe para extraer 33 landmarks corporales en cada fotograma del video.
* **Clasificación en Tiempo Real**: Implementa modelos de machine learning para clasificar cinco actividades: caminar hacia la cámara, caminar de regreso, girar, sentarse y ponerse de pie.
* **Extracción de Características**: Normaliza las coordenadas de los landmarks y calcula características cinemáticas como ángulos y velocidades para el entrenamiento de los modelos.
* **Visualización de Datos**: Interfaz gráfica simple que muestra el video en vivo, la superposición del esqueleto detectado y la etiqueta de la actividad clasificada.

## Metodología

El desarrollo del proyecto sigue un enfoque estructurado e iterativo basado en la metodología **CRISP-DM (Cross-Industry Standard Process for Data Mining)**, cubriendo las fases de comprensión del problema, comprensión y preparación de los datos, modelado, evaluación y despliegue.

## Tecnologías y Librerías

* **Lenguaje**: Python 3.x
* **Visión por Computadora**: OpenCV, MediaPipe
* **Análisis de Datos y Machine Learning**: Pandas, NumPy, Scikit-learn (SVM, RandomForest, XGBoost)
* **Visualización**: Matplotlib, Seaborn

## Instalación y Configuración

Clonar el repositorio:

    ```
    git clone [url_repo)
    cd Sistema-Anotacion-Video
    ```
