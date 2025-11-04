"""
Aplicación de tiempo real para clasificación de actividades humanas
Interfaz gráfica con Streamlit
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Importar módulos propios
from video_processor import VideoProcessor
from activity_classifier import ActivityClassifier


# Configuración de la página
st.set_page_config(
    page_title="Sistema de Clasificación de Actividades",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .activity-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-box {
        background-color: #4CAF50;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Carga los modelos y procesadores (cacheo para eficiencia)"""
    try:
        # Detectar si estamos ejecutando desde Entrega3/ o desde Entrega3/real_time/
        current_dir = Path.cwd()
        
        if current_dir.name == "real_time":
            # Ejecutando desde real_time/
            base_path = current_dir.parent.parent / "Entrega2"
        elif current_dir.name == "Entrega3":
            # Ejecutando desde Entrega3/ (RECOMENDADO)
            base_path = current_dir.parent / "Entrega2"
        else:
            # Fallback: usar ruta del archivo
            base_path = Path(__file__).parent.parent.parent / "Entrega2"
        
        model_path = base_path / "models" / "best_model.pkl"
        scaler_path = base_path / "models" / "scaler.pkl"
        label_encoder_path = base_path / "models" / "label_encoder.pkl"
        
        # Verificar que existan
        missing_files = []
        if not model_path.exists():
            missing_files.append(f"best_model.pkl ({model_path})")
        if not scaler_path.exists():
            missing_files.append(f"scaler.pkl ({scaler_path})")
        if not label_encoder_path.exists():
            missing_files.append(f"label_encoder.pkl ({label_encoder_path})")
        
        if missing_files:
            st.error(" **No se encontraron los modelos entrenados:**")
            for file in missing_files:
                st.error(f"   - {file}")
            st.info("💡 **Solución:** Ejecuta primero:\n```bash\ncd Entrega2/notebooks\npython 04_model_training_gridsearch.py\n```")
            return None, None
        
        # Inicializar procesador y clasificador
        video_processor = VideoProcessor()
        classifier = ActivityClassifier(
            str(model_path),
            str(scaler_path),
            str(label_encoder_path)
        )
        
        return video_processor, classifier
        
    except Exception as e:
        st.error(f" **Error al cargar modelos:** {str(e)}")
        st.exception(e)
        return None, None


def create_probability_chart(probabilities: dict) -> go.Figure:
    """Crea un gráfico de barras con las probabilidades"""
    df = pd.DataFrame({
        'Actividad': list(probabilities.keys()),
        'Probabilidad': list(probabilities.values())
    })
    df = df.sort_values('Probabilidad', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['Probabilidad'],
        y=df['Actividad'],
        orientation='h',
        marker=dict(
            color=df['Probabilidad'],
            colorscale='Viridis',
            showscale=True
        ),
        text=[f"{p:.1%}" for p in df['Probabilidad']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Probabilidades por Actividad",
        xaxis_title="Probabilidad",
        yaxis_title="Actividad",
        height=400,
        showlegend=False
    )
    
    return fig


def create_angles_chart(angles: dict) -> go.Figure:
    """Crea un gráfico de ángulos corporales"""
    df = pd.DataFrame({
        'Articulación': list(angles.keys()),
        'Ángulo (°)': list(angles.values())
    })
    
    fig = go.Figure(go.Bar(
        x=df['Articulación'],
        y=df['Ángulo (°)'],
        marker=dict(
            color=df['Ángulo (°)'],
            colorscale='RdYlGn',
            showscale=True
        ),
        text=[f"{a:.1f}°" for a in df['Ángulo (°)']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Ángulos Corporales en Tiempo Real",
        xaxis_title="Articulación",
        yaxis_title="Ángulo (°)",
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig


def create_timeline_chart(history: list) -> go.Figure:
    """Crea un gráfico de línea temporal de actividades"""
    if not history:
        return None
    
    df = pd.DataFrame(history)
    
    fig = go.Figure()
    
    # Agregar línea de confianza
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['confidence'],
        mode='lines+markers',
        name='Confianza',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Confianza de Predicciones en el Tiempo",
        xaxis_title="Frame",
        yaxis_title="Confianza",
        height=300,
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def main():
    """Función principal de la aplicación"""
    
    # Header
    st.markdown('<div class="main-header">🎥 Sistema de Clasificación de Actividades en Tiempo Real</div>', 
                unsafe_allow_html=True)
    
    # Sidebar con opciones simplificadas
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Opciones de visualización
        st.subheader(" Visualización")
        show_landmarks = st.checkbox("Mostrar landmarks", value=True)
        show_probabilities = st.checkbox("Mostrar probabilidades", value=True)
        
        # Opciones de procesamiento
        st.subheader("🔧 Procesamiento")
        confidence_threshold = st.slider("Umbral de confianza", 0.0, 1.0, 0.5, 0.05)
        
        st.markdown("---")
        st.info("💡 La cámara está siempre activa y analizando movimientos en tiempo real")
    
    # Cargar modelos
    video_processor, classifier = load_models()
    
    if video_processor is None or classifier is None:
        st.warning("️ **Los modelos no pudieron cargarse.**")
        st.info("""
        **Solución:**
        1. Verifica que ejecutaste el entrenamiento:
           ```bash
           cd Entrega2/notebooks
           python 04_model_training_gridsearch.py
           ```
        
        2. O ejecuta la verificación:
           ```bash
           cd Entrega3
           python check_setup.py
           ```
        """)
        return
    
    # Configurar permisos de cámara en macOS
    import os
    os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'
    
    # Inicializar cámara
    if 'cap' not in st.session_state:
        st.session_state.cap = cv2.VideoCapture(0)
        if not st.session_state.cap.isOpened():
            st.error(" No se pudo abrir la cámara")
            st.info("""
            **Solución para cámara en macOS:**
            1. Abre **Preferencias del Sistema** → **Seguridad y Privacidad**
            2. Ve a la pestaña **Privacidad** → **Cámara**
            3. Asegúrate de que tu terminal o Python tenga permisos habilitados
            """)
            return
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video en Tiempo Real")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader(" Predicción Actual")
        activity_placeholder = st.empty()
        confidence_placeholder = st.empty()
    
    # Visualización de probabilidades
    if show_probabilities:
        prob_placeholder = st.empty()
    
    # Botón para control del loop
    stop_button = st.button("🛑 Detener Cámara")
    
    # Loop continuo REAL como en el notebook de ejemplo
    # Este loop se mantiene ejecutando hasta que se presione detener
    frame_count = 0
    max_frames = 1000  # Límite de seguridad
    
    while frame_count < max_frames:
        try:
            # Capturar frame (como en el notebook)
            ret, frame = st.session_state.cap.read()
            
            if not ret:
                st.error(" Error al leer de la cámara")
                break
            
            frame_count += 1
            
            # Extraer landmarks
            result = video_processor.extract_landmarks(frame)
            
            if result:
                landmarks_array = np.array(result['landmarks'])
                
                # Extraer features
                features = video_processor.extract_features_from_landmarks(landmarks_array)
                
                # Clasificar
                prediction = classifier.predict_with_metadata(features)
                
                # Crear frame anotado (copia como en el notebook)
                annotated_frame = frame.copy()
                
                # Dibujar landmarks si está habilitado
                if show_landmarks:
                    annotated_frame = video_processor.draw_landmarks(annotated_frame, result['results'])
                
                # Agregar información al frame (como en el notebook)
                if prediction['confidence'] >= confidence_threshold:
                    activity_text = f"Actividad: {prediction['class']}"
                    confidence_text = f"Confianza: {prediction['confidence']:.1%}"
                    
                    cv2.putText(annotated_frame, activity_text, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, confidence_text, (10, 70),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Convertir a RGB para Streamlit
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Actualizar video (como cv2.imshow pero con Streamlit)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Actualizar predicción
                if prediction['confidence'] >= confidence_threshold:
                    activity_placeholder.markdown(
                        f'<div class="activity-box">{prediction["class"]}</div>',
                        unsafe_allow_html=True
                    )
                    confidence_placeholder.markdown(
                        f'<div class="confidence-box">Confianza: {prediction["confidence"]:.1%}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    activity_placeholder.warning("️ Confianza baja - No detectado")
                
                # Actualizar gráfico de probabilidades
                if show_probabilities:
                    prob_chart = create_probability_chart(prediction['probabilities'])
                    prob_placeholder.plotly_chart(prob_chart, use_container_width=True, key=f"prob_{frame_count}")
            
            else:
                # No se detectó persona (como en el notebook)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                activity_placeholder.warning("️ No se detectó ninguna persona")
            
            # Control de FPS (como cv2.waitKey pero sin bloqueo)
            time.sleep(0.033)  # ~30 FPS
            
            # Forzar actualización de Streamlit cada N frames
            if frame_count % 10 == 0:
                st.empty()  # Pequeño hack para forzar refresh
            
        except Exception as e:
            st.error(f"Error en el procesamiento: {e}")
            break
    
    # Limpiar recursos
    st.info("ℹ️ Sesión finalizada. Refresca la página para iniciar de nuevo.")


if __name__ == "__main__":
    main()
