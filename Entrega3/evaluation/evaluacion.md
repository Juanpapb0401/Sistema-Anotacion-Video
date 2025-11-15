"""
README - Scripts de Evaluación para Entrega 3

Este directorio contiene scripts específicos para analizar y evaluar
los aspectos clave de la Entrega 3 del proyecto.

IMPORTANTE: Estos scripts SON DIFERENTES a la evaluación de Entrega 2.
La evaluación de Entrega 2 ya está completa en:
  - Entrega2/notebooks/05_evaluation.py
  - Entrega2/data/evaluation_results.json

=============================================================================
SCRIPTS DE EVALUACIÓN - ENTREGA 3
=============================================================================

 01_analyze_feature_reduction.py
-----------------------------------
Propósito:
  Analiza el impacto de la reducción de características de 147 → 15 features

¿Qué hace?:
   Visualiza la importancia de las 15 features seleccionadas
   Muestra la importancia acumulada
   Clasifica features por tipo (ángulos, distancias, coordenadas)
   Compara el impacto dimensional (memoria, velocidad)
   Genera reporte detallado de la reducción

Cómo ejecutar:
  ```bash
  cd Entrega3/evaluation
  python 01_analyze_feature_reduction.py
  ```

Outputs:
  - Gráficos:
    * 01_feature_importance_analysis.png
    * 02_feature_types_distribution.png
    * 03_dimensionality_impact.png
  - Datos:
    * feature_reduction_report.txt
    * feature_reduction_analysis.json

Dependencias de datos:
  - Entrega2/data/selected_features.json
  - Entrega2/data/test.csv
  - Entrega2/models/best_model.pkl


