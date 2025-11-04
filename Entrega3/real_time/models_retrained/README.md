# Modelo Re-entrenado - Solo Grupo Principal

## 📊 Información del Modelo

Este modelo fue entrenado **únicamente** con los videos del grupo principal:
- Joshua
- Juan
- Santiago
- Thomas

Se excluyeron los 27 videos del otro grupo para mejorar la consistencia.

## 🎯 Actividades Detectadas

1. **caminar_hacia_camara**: Caminar hacia la cámara
2. **caminar_de_regreso**: Caminar alejándose de la cámara
3. **girar**: Giro de 180° o 360°
4. **sentarse**: Sentarse en una silla
5. **ponerse_de_pie**: Levantarse desde posición sentada

## 🚀 Cómo Usar

### Ejecutar Interface Real-time

```bash
cd Entrega3/real_time
python real_time_retrained.py
```

### Controles

- `q`: Salir de la aplicación
- `s`: Captura de pantalla

## 📈 Comparación con Modelo Anterior

| Aspecto | Modelo Anterior | Modelo Re-entrenado |
|---------|----------------|---------------------|
| **Datos** | 69 videos (42 + 27 otro grupo) | 42 videos (solo grupo) |
| **Features** | 147 (complejas) | 8 (simplificadas) |
| **Enfoque** | Frame-by-frame | Ventanas 0.5 seg |
| **Problema** | Sesgo hacia "caminar" | ¿Más balanceado? |

## 🔍 Features Utilizadas

El modelo usa solo 8 features agregadas estadísticamente:

1. **left_knee_angle_mean**: Promedio ángulo rodilla izquierda
2. **left_knee_angle_std**: Desviación ángulo rodilla izquierda
3. **right_knee_angle_mean**: Promedio ángulo rodilla derecha
4. **right_knee_angle_std**: Desviación ángulo rodilla derecha
5. **trunk_incl_mean**: Promedio inclinación del tronco
6. **trunk_incl_std**: Desviación inclinación del tronco
7. **hip_shoulder_dist_mean**: Promedio distancia caderas-hombros
8. **hip_shoulder_dist_std**: Desviación distancia caderas-hombros

## ⚙️ Configuración

- **Window size**: 15 frames (~0.5 segundos @ 30 FPS)
- **Overlap**: 50%
- **Confidence threshold**: 40%
- **Modelo**: XGBoost (200 estimators, max_depth=8)

## 📝 Archivos

```
models_retrained/
├── xgboost_model.pkl          # Modelo XGBoost entrenado
├── random_forest_model.pkl    # Modelo Random Forest (alternativo)
├── scaler.pkl                 # StandardScaler para normalización
├── label_encoder.pkl          # Codificador de etiquetas
├── model_metadata.json        # Metadatos y métricas del modelo
└── README.md                  # Este archivo
```

## 🎓 Notas Técnicas

### Por qué este enfoque es mejor:

1. **Datos consistentes**: Un solo grupo con estilo uniforme
2. **Features robustas**: Estadísticas agregadas reducen ruido
3. **Ventanas temporales**: Capturan contexto de movimiento
4. **Menos overfitting**: 8 features vs 147 del modelo anterior

### Validación:

- Split: 70% train, 15% val, 15% test
- Estratificación para mantener balance de clases
- Evaluación con F1-score macro (importante para clases desbalanceadas)

## 🐛 Troubleshooting

### El modelo no detecta correctamente:

1. Asegúrate de estar completamente visible en la cámara
2. Mantén la actividad por al menos 0.5 segundos
3. Verifica que el buffer esté lleno (15/15 frames)
4. Prueba con buena iluminación

### Error al cargar modelos:

```bash
# Verifica que existan los archivos
ls models_retrained/
```

## 📞 Contacto

Equipo: Joshua, Juan, Santiago, Thomas
Proyecto: Sistema de Anotación de Video - Entrega 3
