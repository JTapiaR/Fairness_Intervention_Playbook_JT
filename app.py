import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="Plataforma de Equidad en IA",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .step-container {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f7fafc, #edf2f7);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #e2e8f0;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
    }
    
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
    
    .success-box {
        background: rgba(72, 187, 120, 0.1);
        border-left: 4px solid #48bb78;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar estado de sesión
if 'selections' not in st.session_state:
    st.session_state.selections = {}
if 'causal_relations' not in st.session_state:
    st.session_state.causal_relations = []

# Sidebar para navegación
st.sidebar.title("🎯 Equidad en IA")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["🎯 Guía de Decisión", "🧪 Pre-procesamiento", "⚙️ In-procesamiento", 
     "📊 Post-procesamiento", "🔗 Análisis Causal"]
)

# Función para generar recomendaciones
def get_recommendation(selections):
    project_stage = selections.get('project_stage')
    bias_type = selections.get('bias_type')
    resources = selections.get('resources')
    
    if project_stage == 'planning' or resources == 'limited_data':
        return {
            'title': "Intervención de Pre-procesamiento",
            'page': "preprocessing",
            'content': """
            **🔧 Estrategia Recomendada:** Modificar los datos antes del entrenamiento
            
            ✓ Resampling causal para balancear representación  
            ✓ Eliminación de variables proxy problemáticas  
            ✓ Generación de datos contrafactuales  
            ✓ Corrección de etiquetas sesgadas  
            
            **💡 Ventaja:** Control total sobre la calidad de los datos y prevención proactiva de sesgos.
            """
        }
    elif project_stage == 'training' or resources == 'limited_model':
        return {
            'title': "Intervención Durante el Procesamiento",
            'page': "inprocessing",
            'content': """
            **🔧 Estrategia Recomendada:** Modificar el algoritmo de entrenamiento
            
            ✓ Debiasing adversarial para aprender representaciones justas  
            ✓ Regularización causal con restricciones de equidad  
            ✓ Optimización multi-objetivo balanceando precisión y equidad  
            ✓ Aprendizaje con restricciones causales  
            
            **💡 Ventaja:** Balance óptimo entre precisión y equidad con control fino.
            """
        }
    else:
        return {
            'title': "Intervención de Post-procesamiento",
            'page': "postprocessing",
            'content': """
            **🔧 Estrategia Recomendada:** Ajustar las predicciones del modelo
            
            ✓ Optimización de umbrales por grupo demográfico  
            ✓ Calibración de probabilidades por grupo  
            ✓ Clasificación con rechazo para casos ambiguos  
            ✓ Transformaciones de puntuación justas  
            
            **💡 Ventaja:** Implementación rápida sin reentrenar el modelo.
            """
        }

# PÁGINA 1: GUÍA DE DECISIÓN
if page == "🎯 Guía de Decisión":
    st.markdown('<div class="main-header"><h1>🎯 Guía de Decisión Inteligente</h1><p>Encuentra la estrategia perfecta para mitigar sesgos en tu modelo de ML</p></div>', unsafe_allow_html=True)
    
    # Paso 1: Etapa del proyecto
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.markdown("### 📋 Paso 1: ¿En qué etapa se encuentra tu proyecto?")
    
    project_stage = st.radio(
        "",
        ["📋 Planificación - Aún estoy diseñando el modelo y preparando los datos",
         "🔧 Entrenamiento - Tengo control sobre el algoritmo de aprendizaje", 
         "🚀 Modelo Desplegado - El modelo ya está entrenado y en producción"],
        key="project_stage_radio"
    )
    
    if project_stage:
        stage_mapping = {
            "📋 Planificación - Aún estoy diseñando el modelo y preparando los datos": "planning",
            "🔧 Entrenamiento - Tengo control sobre el algoritmo de aprendizaje": "training",
            "🚀 Modelo Desplegado - El modelo ya está entrenado y en producción": "deployed"
        }
        st.session_state.selections['project_stage'] = stage_mapping[project_stage]
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Paso 2: Tipo de sesgo
    if 'project_stage' in st.session_state.selections:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown("### 🔍 Paso 2: ¿Qué tipo de sesgo has identificado?")
        
        bias_type = st.radio(
            "",
            ["📚 Sesgo Histórico - Los datos reflejan discriminación pasada",
             "⚖️ Sesgo de Representación - Algunos grupos están subrepresentados",
             "📏 Sesgo de Medición - Las etiquetas o mediciones son inexactas",
             "🤖 Sesgo Algorítmico - El algoritmo amplifica sesgos existentes"],
            key="bias_type_radio"
        )
        
        if bias_type:
            bias_mapping = {
                "📚 Sesgo Histórico - Los datos reflejan discriminación pasada": "historical",
                "⚖️ Sesgo de Representación - Algunos grupos están subrepresentados": "representation",
                "📏 Sesgo de Medición - Las etiquetas o mediciones son inexactas": "measurement",
                "🤖 Sesgo Algorítmico - El algoritmo amplifica sesgos existentes": "algorithmic"
            }
            st.session_state.selections['bias_type'] = bias_mapping[bias_type]
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Paso 3: Recursos
    if 'bias_type' in st.session_state.selections:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown("### 🛠️ Paso 3: ¿Qué recursos y restricciones tienes?")
        
        resources = st.radio(
            "",
            ["🎛️ Control Total - Puedo modificar datos, algoritmo y predicciones",
             "📊 Solo Datos - Solo puedo modificar los datos de entrenamiento",
             "⚙️ Solo Algoritmo - Puedo modificar el proceso de entrenamiento",
             "📤 Solo Salidas - Solo puedo ajustar las predicciones finales"],
            key="resources_radio"
        )
        
        if resources:
            resources_mapping = {
                "🎛️ Control Total - Puedo modificar datos, algoritmo y predicciones": "full_control",
                "📊 Solo Datos - Solo puedo modificar los datos de entrenamiento": "limited_data",
                "⚙️ Solo Algoritmo - Puedo modificar el proceso de entrenamiento": "limited_model",
                "📤 Solo Salidas - Solo puedo ajustar las predicciones finales": "limited_output"
            }
            st.session_state.selections['resources'] = resources_mapping[resources]
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Mostrar recomendación
    if len(st.session_state.selections) == 3:
        rec = get_recommendation(st.session_state.selections)
        st.markdown(f'<div class="recommendation-box"><h3>{rec["title"]}</h3>{rec["content"]}</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Empezar de Nuevo"):
            st.session_state.selections = {}
            st.rerun()

# PÁGINA 2: PRE-PROCESAMIENTO
elif page == "🧪 Pre-procesamiento":
    st.markdown('<div class="main-header"><h1>🧪 Toolkit de Pre-procesamiento</h1><p>Modifica y mejora tus datos antes del entrenamiento</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box"><strong>🔍 ¿Qué es el Pre-procesamiento?</strong><br>Consiste en "limpiar" los datos antes de que el modelo aprenda de ellos. Es como preparar los ingredientes para una receta: si algunos están sesgados, los ajustas antes de cocinar.</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Representación", "🔍 Correlaciones", "⚖️ Re-muestreo", "🎮 Simulación"])
    
    with tab1:
        st.markdown("### 📊 Análisis de Representación")
        st.markdown('<div class="info-box">Verifica si todos los grupos demográficos están representados de manera justa en tus datos.</div>', unsafe_allow_html=True)
        
        data_a = st.slider("Grupo A en tus datos (%)", 0, 100, 70)
        data_b = 100 - data_a
        pop_a, pop_b = 50, 50
        
        gap = abs(data_a - pop_a)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[
                go.Bar(name='Población Real', x=['Grupo A', 'Grupo B'], y=[pop_a, pop_b]),
                go.Bar(name='Tus Datos', x=['Grupo A', 'Grupo B'], y=[data_a, data_b])
            ])
            fig.update_layout(title='Comparación de Representación', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><h2 style="color: #667eea;">{gap}%</h2><p>Brecha de Representación</p></div>', unsafe_allow_html=True)
        
        if gap > 10:
            st.markdown(f'<div class="warning-box">Hay una brecha de representación significativa. El Grupo A está {"sobre" if data_a > pop_a else "sub"}representado en {gap} puntos porcentuales.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">La representación en tus datos es similar a la población de referencia.</div>', unsafe_allow_html=True)
        
        st.text_area("Documenta tu análisis de representación:", key="representation_notes")
    
    with tab2:
        st.markdown("### 🔍 Detección de Variables Proxy")
        st.markdown('<div class="info-box">Identifica variables aparentemente neutrales que estén correlacionadas con atributos protegidos.</div>', unsafe_allow_html=True)
        
        # Generar datos sintéticos para demostrar correlación proxy
        np.random.seed(42)
        groups = np.random.choice([0, 1], 100)
        proxy_values = groups * 20 + np.random.normal(0, 5, 100) + 50
        outcomes = proxy_values * 3 + np.random.normal(0, 20, 100) + 200
        
        col1, col2 = st.columns(2)
        
        with col1:
            df_proxy = pd.DataFrame({
                'Grupo': groups,
                'Proxy': proxy_values
            })
            fig1 = px.box(df_proxy, x='Grupo', y='Proxy', title='Grupo vs. Variable Proxy')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            df_outcome = pd.DataFrame({
                'Proxy': proxy_values,
                'Resultado': outcomes,
                'Grupo': groups
            })
            fig2 = px.scatter(df_outcome, x='Proxy', y='Resultado', color='Grupo', 
                             title='Variable Proxy vs. Resultado')
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown('<div class="info-box">El código postal está correlacionado tanto con el grupo demográfico como con el resultado, haciéndolo un proxy problemático.</div>', unsafe_allow_html=True)
        st.text_area("Documenta las variables proxy identificadas:", key="proxy_notes")
    
    with tab3:
        st.markdown("### ⚖️ Técnicas de Re-muestreo")
        st.markdown('<div class="info-box">Equilibra la representación mediante sobremuestreo (duplicar muestras minoritarias) o submuestreo (reducir muestras mayoritarias).</div>', unsafe_allow_html=True)
        
        factor = st.slider("Factor de sobremuestreo", 1, 5, 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_before = go.Figure(data=[
                go.Bar(x=['Grupo A', 'Grupo B'], y=[100, 20], name='Antes')
            ])
            fig_before.update_layout(title='Datos Originales')
            st.plotly_chart(fig_before, use_container_width=True)
        
        with col2:
            fig_after = go.Figure(data=[
                go.Bar(x=['Grupo A', 'Grupo B'], y=[100, 20 * factor], name='Después')
            ])
            fig_after.update_layout(title='Después del Sobremuestreo')
            st.plotly_chart(fig_after, use_container_width=True)
        
        st.text_area("Documenta tu estrategia de re-muestreo:", key="resampling_notes")
    
    with tab4:
        st.markdown("### 🎮 Simulación Interactiva: Preprocesamiento")
        
        bias_level = st.slider("Sesgo en los datos (%)", 0, 100, 50)
        correction_level = st.slider("Intensidad de corrección (%)", 0, 100, 0)
        
        improvement = (correction_level / 100) * (bias_level / 100) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            original_bias = bias_level
            corrected_bias = bias_level * (1 - correction_level / 100)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=['Original', 'Después del Preprocesamiento'],
                y=[original_bias, corrected_bias],
                mode='lines+markers',
                name='Nivel de Sesgo',
                line=dict(color='#fc8181', width=3)
            ))
            fig.update_layout(title='Efecto del Preprocesamiento en el Sesgo')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><h2 style="color: #667eea;">{improvement:.0f}%</h2><p>Mejora en Equidad</p></div>', unsafe_allow_html=True)

        # Comparación de técnicas usando radar chart
        st.markdown("### 📊 Comparación de Técnicas de Pre-procesamiento")
        techniques = ['Resampling', 'Proxy Removal', 'Data Augmentation']
        metrics = ['Precisión', 'Equidad', 'Interpretabilidad', 'Velocidad', 'Estabilidad']
        
        # Datos para el radar chart
        data = {
            'Resampling': [80, 85, 90, 95, 85],
            'Proxy Removal': [90, 80, 95, 85, 90],
            'Data Augmentation': [85, 90, 75, 70, 80]
        }
        
        fig = go.Figure()
        
        for technique in techniques:
            fig.add_trace(go.Scatterpolar(
                r=data[technique],
                theta=metrics,
                fill='toself',
                name=technique
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            title="Comparación de Técnicas de Pre-procesamiento"
        )
        st.plotly_chart(fig, use_container_width=True)

# PÁGINA 3: IN-PROCESAMIENTO
elif page == "⚙️ In-procesamiento":
    st.markdown('<div class="main-header"><h1>⚙️ Toolkit de In-procesamiento</h1><p>Modifica el algoritmo de entrenamiento para incluir equidad</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box"><strong>🔍 ¿Qué es el In-procesamiento?</strong><br>Modifica el algoritmo de aprendizaje para que la equidad sea uno de sus objetivos, junto con la precisión.</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎭 Debiasing Adversario", "⚖️ Restricciones", "🎯 Multi-objetivo", "🎮 Simulación"])
    
    with tab1:
        st.markdown("### 🎭 Debiasing Adversario")
        st.markdown('<div class="info-box">Un juego entre dos IAs: un Predictor que hace su trabajo y un Adversario que intenta detectar sesgos. El Predictor aprende a ser justo para engañar al Adversario.</div>', unsafe_allow_html=True)
        
        st.markdown("#### 🏗️ Arquitectura Adversarial")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Crear diagrama de arquitectura usando texto ASCII
            st.markdown("""
            ```
            Datos
              ↓
            Predictor ← Adversario
              ↓             ↑
            Predicción  Detección Sesgo
            ```
            """)
        
        st.markdown('<div class="info-box">El Adversario intenta detectar el grupo demográfico desde las predicciones. El Predictor aprende a hacer predicciones que sean útiles pero que no revelen información demográfica.</div>', unsafe_allow_html=True)
        
        # Simulación del entrenamiento adversarial
        epochs = st.slider("Épocas de entrenamiento", 1, 100, 50)
        
        # Simular curvas de aprendizaje
        np.random.seed(42)
        predictor_loss = 1.0 * np.exp(-np.arange(epochs) / 30) + 0.1 + np.random.normal(0, 0.02, epochs)
        adversary_loss = 0.7 * np.exp(-np.arange(epochs) / 40) + 0.5 + np.random.normal(0, 0.02, epochs)
        fairness_score = 1 - adversary_loss  # A menor pérdida del adversario, menor equidad
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=list(range(epochs)), y=predictor_loss, name="Pérdida Predictor"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=list(range(epochs)), y=fairness_score, name="Puntuación Equidad"),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Épocas")
        fig.update_yaxes(title_text="Pérdida Predictor", secondary_y=False)
        fig.update_yaxes(title_text="Puntuación Equidad", secondary_y=True)
        fig.update_layout(title_text="Entrenamiento Adversarial")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.text_area("Describe tu arquitectura adversarial:", key="adversarial_notes")
    
    with tab2:
        st.markdown("### ⚖️ Restricciones de Equidad")
        st.markdown('<div class="info-box">Incorpora "reglas de equidad" directamente en las matemáticas del modelo usando multiplicadores de Lagrange.</div>', unsafe_allow_html=True)
        
        st.markdown("#### 📐 Fórmula de Lagrange")
        st.latex(r"L(\theta, \lambda) = \text{Pérdida Original}(\theta) + \lambda \times \text{Restricción Equidad}(\theta)")
        
        # Selector de tipo de restricción
        constraint_type = st.selectbox(
            "Tipo de restricción de equidad:",
            ["Paridad Demográfica", "Igualdad de Oportunidades", "Odds Ecualizadas", "Calibración"]
        )
        
        lambda_val = st.slider("Multiplicador de Lagrange (λ)", 0.0, 2.0, 0.5, 0.1)
        
        # Mostrar el efecto del multiplicador
        col1, col2 = st.columns(2)
        
        with col1:
            # Simular pérdida total
            original_loss = 0.3
            fairness_penalty = lambda_val * 0.2
            total_loss = original_loss + fairness_penalty
            
            fig = go.Figure(data=[
                go.Bar(name='Pérdida Original', x=['Modelo'], y=[original_loss]),
                go.Bar(name='Penalización Equidad', x=['Modelo'], y=[fairness_penalty])
            ])
            fig.update_layout(title='Componentes de la Pérdida Total', barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><h2 style="color: #667eea;">{constraint_type}</h2><p>Restricción Activa</p></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><h2 style="color: #48bb78;">{total_loss:.2f}</h2><p>Pérdida Total</p></div>', unsafe_allow_html=True)
        
        st.text_area("Define tus restricciones de equidad:", key="constraints_notes")
    
    with tab3:
        st.markdown("### 🎯 Optimización Multi-objetivo")
        st.markdown('<div class="info-box">Trata la precisión y la equidad como objetivos separados que deben equilibrarse en la frontera de Pareto.</div>', unsafe_allow_html=True)
        
        # Generar puntos de la frontera de Pareto
        np.random.seed(42)
        accuracy = np.linspace(0.80, 0.95, 20)
        fairness = 1 - np.sqrt(accuracy - 0.79) + np.random.normal(0, 0.02, 20)
        fairness = np.clip(fairness, 0.5, 1.0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=accuracy,
            y=fairness,
            mode='markers',
            marker=dict(
                size=10,
                color=accuracy + fairness,
                colorscale='Viridis',
                showscale=True
            ),
            text=[f'Modelo {i+1}' for i in range(len(accuracy))],
            name='Modelos Posibles'
        ))
        fig.update_layout(
            title='Frontera de Pareto: Precisión vs. Equidad',
            xaxis_title='Precisión del Modelo',
            yaxis_title='Puntuación de Equidad'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Selector de modelo
        selected_model = st.selectbox("Selecciona un modelo:", [f"Modelo {i+1}" for i in range(20)])
        model_idx = int(selected_model.split()[1]) - 1
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><h2 style="color: #667eea;">{accuracy[model_idx]*100:.1f}%</h2><p>Precisión</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h2 style="color: #48bb78;">{fairness[model_idx]*100:.1f}%</h2><p>Equidad</p></div>', unsafe_allow_html=True)
        
        st.text_area("Define tus múltiples objetivos:", key="multiobjective_notes")
    
    with tab4:
        st.markdown("### 🎮 Simulación: Trade-off Precisión-Equidad")
        
        lambda_weight = st.slider("Peso de la equidad (λ)", 0.0, 1.0, 0.5, 0.1)
        
        # Simular trade-off
        base_accuracy = 0.85
        base_fairness = 0.60
        
        accuracy = base_accuracy * (1 - lambda_weight * 0.15)
        fairness = base_fairness + lambda_weight * 0.35
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'<div class="metric-card"><h2 style="color: #667eea;">{accuracy*100:.0f}%</h2><p>Precisión del Modelo</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><h2 style="color: #48bb78;">{fairness*100:.0f}%</h2><p>Puntuación de Equidad</p></div>', unsafe_allow_html=True)
        
        if lambda_weight < 0.3:
            st.markdown('<div class="info-box">Priorizando precisión sobre equidad.</div>', unsafe_allow_html=True)
        elif lambda_weight > 0.7:
            st.markdown('<div class="warning-box">Alta prioridad en equidad puede reducir significativamente la precisión.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">Buen equilibrio entre precisión y equidad.</div>', unsafe_allow_html=True)
        
        # Gráfico del trade-off
        lambda_range = np.linspace(0, 1, 11)
        acc_curve = base_accuracy * (1 - lambda_range * 0.15)
        fair_curve = base_fairness + lambda_range * 0.35
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lambda_range, y=acc_curve, mode='lines+markers', name='Precisión'))
        fig.add_trace(go.Scatter(x=lambda_range, y=fair_curve, mode='lines+markers', name='Equidad'))
        fig.add_vline(x=lambda_weight, line_dash="dash", line_color="red", annotation_text=f"λ actual: {lambda_weight}")
        fig.update_layout(
            title='Trade-off Precisión vs. Equidad',
            xaxis_title='Peso de Equidad (λ)',
            yaxis_title='Puntuación'
        )
        st.plotly_chart(fig, use_container_width=True)

# PÁGINA 4: POST-PROCESAMIENTO
elif page == "📊 Post-procesamiento":
    st.markdown('<div class="main-header"><h1>📊 Toolkit de Post-procesamiento</h1><p>Ajusta las predicciones después del entrenamiento</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box"><strong>🔍 ¿Qué es el Post-procesamiento?</strong><br>Ajusta las predicciones de un modelo después de que ya ha sido entrenado, como un editor que revisa un texto para corregir sesgos.</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Umbrales", "📏 Calibración", "🚫 Rechazo", "🎮 Simulación"])
    
    with tab1:
        st.markdown("### 🎯 Optimización de Umbrales")
        st.markdown('<div class="info-box">Ajusta los umbrales de decisión para diferentes grupos para lograr igualdad de oportunidades.</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            thresh_a = st.slider("Umbral Grupo A", 0.0, 1.0, 0.5, 0.01)
        
        with col2:
            thresh_b = st.slider("Umbral Grupo B", 0.0, 1.0, 0.5, 0.01)
        
        # Simular TPRs
        tpr_a = max(0, min(1, 0.5 + (0.7 - thresh_a) * 1.5))
        tpr_b = max(0, min(1, 0.4 + (0.6 - thresh_b) * 1.5))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'<div class="metric-card"><h2 style="color: #667eea;">{tpr_a*100:.0f}%</h2><p>TPR Grupo A</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><h2 style="color: #fc8181;">{tpr_b*100:.0f}%</h2><p>TPR Grupo B</p></div>', unsafe_allow_html=True)
        
        diff = abs(tpr_a - tpr_b)
        if diff < 0.02:
            st.markdown(f'<div class="success-box">¡Excelente! Has logrado igualdad de oportunidades. Diferencia de TPR: {diff*100:.1f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="warning-box">Ajusta los umbrales para igualar las TPRs. Diferencia actual: {diff*100:.1f}%</div>', unsafe_allow_html=True)
        
        # Visualización de las distribuciones y umbrales
        x = np.linspace(0, 1, 100)
        # Distribuciones simuladas para cada grupo
        dist_a = np.exp(-((x - 0.6) ** 2) / (2 * 0.15 ** 2))
        dist_b = np.exp(-((x - 0.4) ** 2) / (2 * 0.15 ** 2))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=dist_a, mode='lines', name='Distribución Grupo A', fill='tonexty'))
        fig.add_trace(go.Scatter(x=x, y=dist_b, mode='lines', name='Distribución Grupo B', fill='tonexty'))
        fig.add_vline(x=thresh_a, line_dash="dash", line_color="blue", annotation_text=f"Umbral A: {thresh_a:.2f}")
        fig.add_vline(x=thresh_b, line_dash="dash", line_color="red", annotation_text=f"Umbral B: {thresh_b:.2f}")
        fig.update_layout(title='Distribuciones de Puntuación y Umbrales', xaxis_title='Puntuación', yaxis_title='Densidad')
        st.plotly_chart(fig, use_container_width=True)
        
        st.text_area("Documenta tu estrategia de umbrales:", key="thresholds_notes")
    
    with tab2:
        st.markdown("### 📏 Calibración por Grupos")
        st.markdown('<div class="info-box">Asegura que una predicción de "80% de probabilidad" signifique lo mismo para todos los grupos.</div>', unsafe_allow_html=True)
        
        x = np.linspace(0, 1, 21)
        perfect = x
        group_a = np.minimum(1, x * 0.8 + 0.1 + np.random.normal(0, 0.05, len(x)))
        group_b = np.minimum(1, x * 1.2 - 0.05 + np.random.normal(0, 0.05, len(x)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=perfect, mode='lines', name='Calibración Perfecta', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=x, y=group_a, mode='lines+markers', name='Grupo A'))
        fig.add_trace(go.Scatter(x=x, y=group_b, mode='lines+markers', name='Grupo B'))
        fig.update_layout(
            title='Curvas de Calibración por Grupo',
            xaxis_title='Probabilidad Predicha',
            yaxis_title='Fracción Real de Positivos'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas de calibración
        col1, col2, col3 = st.columns(3)
        
        # Calcular error de calibración esperado (ECE) simulado
        ece_a = np.mean(np.abs(group_a - perfect))
        ece_b = np.mean(np.abs(group_b - perfect))
        ece_overall = (ece_a + ece_b) / 2
        
        with col1:
            st.markdown(f'<div class="metric-card"><h2 style="color: #667eea;">{ece_a:.3f}</h2><p>ECE Grupo A</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><h2 style="color: #fc8181;">{ece_b:.3f}</h2><p>ECE Grupo B</p></div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<div class="metric-card"><h2 style="color: #48bb78;">{ece_overall:.3f}</h2><p>ECE General</p></div>', unsafe_allow_html=True)
        
        st.text_area("Documenta tu plan de calibración:", key="calibration_notes")
    
    with tab3:
        st.markdown("### 🚫 Clasificación con Rechazo")
        st.markdown('<div class="info-box">Identifica casos ambiguos y los envía a revisión humana en lugar de tomar decisiones automáticas potencialmente sesgadas.</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            reject_low = st.slider("Umbral inferior", 0.0, 0.5, 0.25, 0.01)
        
        with col2:
            reject_high = st.slider("Umbral superior", 0.5, 1.0, 0.75, 0.01)
        
        # Simular distribución de decisiones
        np.random.seed(42)
        scores = np.random.uniform(0, 1, 1000)
        auto_low = np.sum(scores <= reject_low)
        rejected = np.sum((scores > reject_low) & (scores < reject_high))
        auto_high = np.sum(scores >= reject_high)
        
        automation_rate = (auto_low + auto_high) / 1000 * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[
                go.Bar(x=['Auto (Baja)', 'Rechazo', 'Auto (Alta)'], 
                      y=[auto_low, rejected, auto_high],
                      marker_color=['#48bb78', '#ffc107', '#667eea'])
            ])
            fig.update_layout(title='Distribución de Decisiones')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><h2 style="color: #667eea;">{automation_rate:.0f}%</h2><p>Tasa de Automatización</p></div>', unsafe_allow_html=True)
        
        # Visualización de la zona de rechazo
        x = np.linspace(0, 1, 1000)
        density = np.ones_like(x)  # Distribución uniforme para simplicidad
        
        fig = go.Figure()
        
        # Área de decisión automática baja
        fig.add_trace(go.Scatter(
            x=x[x <= reject_low], 
            y=density[x <= reject_low],
            fill='tozeroy',
            name='Auto (Rechazar)',
            fillcolor='rgba(72, 187, 120, 0.5)'
        ))
        
        # Área de rechazo
        fig.add_trace(go.Scatter(
            x=x[(x > reject_low) & (x < reject_high)], 
            y=density[(x > reject_low) & (x < reject_high)],
            fill='tozeroy',
            name='Rechazo (Humano)',
            fillcolor='rgba(255, 193, 7, 0.5)'
        ))
        
        # Área de decisión automática alta
        fig.add_trace(go.Scatter(
            x=x[x >= reject_high], 
            y=density[x >= reject_high],
            fill='tozeroy',
            name='Auto (Aceptar)',
            fillcolor='rgba(102, 126, 234, 0.5)'
        ))
        
        fig.update_layout(
            title='Zonas de Decisión Automática vs. Revisión Humana',
            xaxis_title='Puntuación del Modelo',
            yaxis_title='Densidad'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.text_area("Diseña tu sistema de rechazo:", key="rejection_notes")
    
    with tab4:
        st.markdown("### 🎮 Simulación Integrada: Post-procesamiento")
        
        # Comparación de técnicas usando radar chart
        techniques = ['Umbrales', 'Calibración', 'Rechazo']
        metrics = ['Precisión', 'Equidad', 'Interpretabilidad', 'Velocidad', 'Estabilidad']
        
        # Datos para el radar chart
        data = {
            'Umbrales': [85, 90, 95, 98, 90],
            'Calibración': [88, 85, 80, 85, 95],
            'Rechazo': [75, 95, 90, 70, 85]
        }
        
        fig = go.Figure()
        
        for technique in techniques:
            fig.add_trace(go.Scatterpolar(
                r=data[technique],
                theta=metrics,
                fill='toself',
                name=technique
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            title="Comparación de Técnicas de Post-procesamiento"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recomendación basada en los datos
        best_technique = max(data.keys(), key=lambda k: sum(data[k]))
        st.markdown(f'<div class="metric-card"><h2 style="color: #667eea;">{best_technique}</h2><p>Técnica Recomendada</p></div>', unsafe_allow_html=True)
        
        # Simulación interactiva del costo-beneficio
        st.markdown("#### 💰 Análisis Costo-Beneficio")
        
        human_review_cost = st.slider("Costo de revisión humana ($ por caso)", 1, 100, 25)
        automation_savings = st.slider("Ahorro por automatización ($ por caso)", 1, 50, 10)
        
        # Calcular costos para diferentes técnicas
        cases_to_review = {
            'Umbrales': 100,  # Casos que requieren ajuste manual
            'Calibración': 150,  # Casos que necesitan recalibración
            'Rechazo': rejected  # Casos enviados a revisión humana
        }
        
        total_costs = {}
        for technique in techniques:
            review_cost = cases_to_review[technique] * human_review_cost
            automation_benefit = (1000 - cases_to_review[technique]) * automation_savings
            total_costs[technique] = review_cost - automation_benefit
        
        fig = go.Figure(data=[
            go.Bar(x=list(total_costs.keys()), y=list(total_costs.values()),
                   marker_color=['#667eea', '#48bb78', '#fc8181'])
        ])
        fig.update_layout(
            title='Costo Neto por Técnica (menor es mejor)',
            yaxis_title='Costo Neto ($)',
            xaxis_title='Técnica'
        )
        st.plotly_chart(fig, use_container_width=True)

# PÁGINA 5: ANÁLISIS CAUSAL
elif page == "🔗 Análisis Causal":
    st.markdown('<div class="main-header"><h1>🔗 Toolkit de Análisis Causal</h1><p>Entiende las relaciones causales para intervenciones más efectivas</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box"><strong>🔍 ¿Qué es el Análisis Causal?</strong><br>Va más allá de las correlaciones para entender el "porqué" de las disparidades, como un detective que reconstruye la cadena de causa y efecto.</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Identificación", "🔄 Contrafactuales", "📊 Diagramas", "🧮 Inferencia"])
    
    with tab1:
        st.markdown("### 🔍 Identificación de Mecanismos")
        
        with st.expander("1. Discriminación Directa"):
            st.markdown('<div class="info-box">Ocurre cuando un atributo protegido se usa explícitamente para tomar decisiones.</div>', unsafe_allow_html=True)
            
            # Ejemplo visual de discriminación directa
            fig = go.Figure()
            fig.add_trace(go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=["Género", "Decisión Préstamo", "Aprobado", "Rechazado"],
                    color=["blue", "gray", "green", "red"]
                ),
                link=dict(
                    source=[0, 0, 1, 1],
                    target=[2, 3, 2, 3],
                    value=[30, 70, 80, 20],
                    color=["rgba(0,255,0,0.4)", "rgba(255,0,0,0.4)", "rgba(0,255,0,0.4)", "rgba(255,0,0,0.4)"]
                )
            ))
            fig.update_layout(title_text="Ejemplo: Discriminación Directa en Préstamos", font_size=10)
            st.plotly_chart(fig, use_container_width=True)
            
            st.text_area("¿El atributo protegido influye directamente en la decisión?", key="direct_discrimination")
        
        with st.expander("2. Discriminación Indirecta"):
            st.markdown('<div class="info-box">El atributo protegido afecta factores intermedios legítimos, transmitiendo el sesgo.</div>', unsafe_allow_html=True)
            
            # Diagrama de discriminación indirecta
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Género**")
                st.markdown("↓")
                st.markdown("(Afecta)")
            
            with col2:
                st.markdown("**Educación**")
                st.markdown("↓")
                st.markdown("(Influye en)")
            
            with col3:
                st.markdown("**Decisión**")
                st.markdown("✓")
                st.markdown("(Resultado)")
            
            st.markdown('<div class="warning-box">El género no se usa directamente, pero afecta la educación, que sí se usa para decidir.</div>', unsafe_allow_html=True)
            
            st.text_area("¿El atributo protegido afecta factores intermedios legítimos?", key="indirect_discrimination")
        
        with st.expander("3. Discriminación por Proxy"):
            st.markdown('<div class="info-box">Variables aparentemente neutrales que correlacionan fuertemente con atributos protegidos.</div>', unsafe_allow_html=True)
            
            # Matriz de correlación simulada
            variables = ['Código Postal', 'Historial Crediticio', 'Ingresos', 'Grupo Étnico']
            correlation_matrix = np.array([
                [1.0, 0.3, 0.4, 0.8],  # Código Postal
                [0.3, 1.0, 0.6, 0.2],  # Historial Crediticio
                [0.4, 0.6, 1.0, 0.3],  # Ingresos
                [0.8, 0.2, 0.3, 1.0]   # Grupo Étnico
            ])
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix,
                x=variables,
                y=variables,
                colorscale='RdBu',
                zmid=0
            ))
            fig.update_layout(title='Matriz de Correlación - Detección de Proxies')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="warning-box">El código postal está fuertemente correlacionado (0.8) con el grupo étnico, convirtiéndolo en un proxy problemático.</div>', unsafe_allow_html=True)
            
            st.text_area("¿Las decisiones dependen de variables correlacionadas con atributos protegidos?", key="proxy_discrimination")
    
    with tab2:
        st.markdown("### 🔄 Análisis Contrafactual")
        st.markdown('<div class="info-box">Analiza qué habría pasado si solo cambiara el atributo protegido, manteniendo todo lo demás constante.</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Caso Original")
            st.markdown("""
            **Grupo:** B  
            **Puntaje:** 650  
            **Decisión:** Rechazado
            """)
        
        with col2:
            st.markdown("#### Contrafactual")
            st.markdown("""
            **Grupo:** A  
            **Puntaje:** 710 (+60 por cambio de grupo)  
            **Decisión:** Aprobado
            """)
        
        st.markdown('<div class="warning-box">El cambio solo en el grupo demográfico alteró la decisión, indicando sesgo causal.</div>', unsafe_allow_html=True)
        
        # Generador de consultas contrafactuales
        st.markdown("#### 🎮 Generador de Consultas Contrafactuales")
        
        original_group = st.selectbox("Grupo original:", ["A", "B"])
        original_score = st.slider("Puntaje original:", 300, 850, 650)
        
        # Simular efecto contrafactual
        group_effect = 60 if original_group == "B" else -60
        counterfactual_score = original_score + group_effect
        counterfactual_group = "A" if original_group == "B" else "B"
        
        # Umbral de decisión
        threshold = 700
        original_decision = "Aprobado" if original_score >= threshold else "Rechazado"
        counterfactual_decision = "Aprobado" if counterfactual_score >= threshold else "Rechazado"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'<div class="metric-card"><h2 style="color: #667eea;">{original_decision}</h2><p>Decisión Original<br>(Grupo {original_group}, Puntaje {original_score})</p></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="metric-card"><h2 style="color: #fc8181;">{counterfactual_decision}</h2><p>Decisión Contrafactual<br>(Grupo {counterfactual_group}, Puntaje {counterfactual_score})</p></div>', unsafe_allow_html=True)
        
        if original_decision != counterfactual_decision:
            st.markdown('<div class="warning-box">🚨 <strong>Sesgo Causal Detectado:</strong> El cambio de grupo alteró la decisión.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ <strong>Sin Sesgo Causal:</strong> El cambio de grupo no alteró la decisión.</div>', unsafe_allow_html=True)
        
        st.text_area("Documenta tus consultas contrafactuales:", key="counterfactual_notes")
    
    with tab3:
        st.markdown("### 📊 Construcción de Diagramas Causales")
        st.markdown('<div class="info-box">Visualiza las relaciones causales usando Diagramas Acíclicos Dirigidos (DAGs).</div>', unsafe_allow_html=True)
        
        st.markdown("#### 🎨 Constructor de DAG Interactivo")
        
        relations = [
            "Género → Educación",
            "Educación → Ingresos", 
            "Ingresos → Decisión",
            "Género → Decisión"
        ]
        
        selected_relations = st.multiselect(
            "Selecciona las relaciones causales:",
            relations,
            default=st.session_state.causal_relations
        )
        
        st.session_state.causal_relations = selected_relations
        
        if selected_relations:
            st.markdown("#### Tu Diagrama Causal")
            
            # Crear visualización del DAG usando networkx-style layout
            nodes = set()
            edges = []
            
            for relation in selected_relations:
                from_node, to_node = relation.split(" → ")
                nodes.add(from_node)
                nodes.add(to_node)
                edges.append((from_node, to_node))
            
            # Posiciones fijas para los nodos comunes
            node_positions = {
                "Género": (0, 1),
                "Educación": (1, 1.5),
                "Ingresos": (2, 1.5),
                "Decisión": (3, 1)
            }
            
            # Crear el gráfico
            fig = go.Figure()
            
            # Añadir edges
            for from_node, to_node in edges:
                if from_node in node_positions and to_node in node_positions:
                    x0, y0 = node_positions[from_node]
                    x1, y1 = node_positions[to_node]
                    
                    fig.add_trace(go.Scatter(
                        x=[x0, x1], y=[y0, y1],
                        mode='lines',
                        line=dict(color='gray', width=2),
                        showlegend=False,
                        hoverinfo='none'
                    ))
                    
                    # Añadir flecha
                    fig.add_annotation(
                        x=x1, y=y1,
                        ax=x0, ay=y0,
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor='gray',
                        showarrow=True
                    )
            
            # Añadir nodes
            for node in nodes:
                if node in node_positions:
                    x, y = node_positions[node]
                    fig.add_trace(go.Scatter(
                        x=[x], y=[y],
                        mode='markers+text',
                        marker=dict(size=30, color='lightblue', line=dict(width=2, color='darkblue')),
                        text=[node],
                        textposition='middle center',
                        showlegend=False,
                        hoverinfo='text'
                    ))
            
            fig.update_layout(
                title="Diagrama Causal (DAG)",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Análisis del DAG
            st.markdown("#### 🔍 Análisis de tu DAG")
            
            if "Género → Decisión" in selected_relations:
                st.markdown('<div class="warning-box">⚠️ <strong>Discriminación Directa Detectada:</strong> Existe un camino directo de Género a Decisión.</div>', unsafe_allow_html=True)
            
            # Detectar caminos indirectos
            indirect_paths = []
            if "Género → Educación" in selected_relations and "Educación → Ingresos" in selected_relations and "Ingresos → Decisión" in selected_relations:
                indirect_paths.append("Género → Educación → Ingresos → Decisión")
            if "Género → Educación" in selected_relations and "Educación → Decisión" in selected_relations:
                indirect_paths.append("Género → Educación → Decisión")
            
            if indirect_paths:
                st.markdown(f'<div class="info-box">🔗 <strong>Caminos Indirectos Encontrados:</strong><br>{"<br>".join(indirect_paths)}</div>', unsafe_allow_html=True)
            
            if st.button("🗑️ Limpiar DAG"):
                st.session_state.causal_relations = []
                st.rerun()
        else:
            st.info("Selecciona relaciones para construir tu DAG")
        
        st.text_area("Documenta tus supuestos causales:", key="causal_assumptions")
    
    with tab4:
        st.markdown("### 🧮 Métodos de Inferencia Causal")
        
        with st.expander("🎯 Emparejamiento (Matching)"):
            st.markdown('<div class="info-box">Compara individuos "gemelos" de diferentes grupos para aislar efectos causales.</div>', unsafe_allow_html=True)
            
            # Generar datos sintéticos para matching
            np.random.seed(42)
            treatment_x = np.random.normal(5, 1, 50)
            treatment_y = np.random.normal(70, 10, 50)
            control_x = np.random.normal(3, 1, 50)
            control_y = np.random.normal(50, 10, 50)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_before = go.Figure()
                fig_before.add_trace(go.Scatter(x=treatment_x, y=treatment_y, mode='markers', name='Tratamiento', marker_color='#fc8181'))
                fig_before.add_trace(go.Scatter(x=control_x, y=control_y, mode='markers', name='Control', marker_color='#667eea'))
                fig_before.update_layout(title='Antes del Matching', xaxis_title='Característica X', yaxis_title='Resultado Y')
                st.plotly_chart(fig_before, use_container_width=True)
            
            with col2:
                # Simular matching tomando subconjunto más cercano
                matched_treatment_x = treatment_x[:25]
                matched_treatment_y = treatment_y[:25]
                # Ajustar control para que sea más similar
                matched_control_x = control_x + 1  # Acercar las distribuciones
                matched_control_y = control_y + 10
                
                fig_after = go.Figure()
                fig_after.add_trace(go.Scatter(x=matched_treatment_x, y=matched_treatment_y, mode='markers', name='Tratamiento (Emparejado)', marker_color='#fc8181'))
                fig_after.add_trace(go.Scatter(x=matched_control_x, y=matched_control_y, mode='markers', name='Control (Emparejado)', marker_color='#667eea'))
                fig_after.update_layout(title='Después del Matching', xaxis_title='Característica X', yaxis_title='Resultado Y')
                st.plotly_chart(fig_after, use_container_width=True)
            
            st.markdown('<div class="info-box">Izquierda: grupos no comparables. Derecha: después del emparejamiento, más comparable.</div>', unsafe_allow_html=True)
            
            # Calcular efecto del tratamiento
            treatment_effect_before = np.mean(treatment_y) - np.mean(control_y)
            treatment_effect_after = np.mean(matched_treatment_y) - np.mean(matched_control_y)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="metric-card"><h2 style="color: #667eea;">{treatment_effect_before:.1f}</h2><p>Efecto Antes del Matching</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h2 style="color: #48bb78;">{treatment_effect_after:.1f}</h2><p>Efecto Después del Matching</p></div>', unsafe_allow_html=True)
        
        with st.expander("📏 Regresión por Discontinuidad"):
            st.markdown('<div class="info-box">Aprovecha umbrales naturales para estimar efectos causales comparando casos justo arriba y abajo del corte.</div>', unsafe_allow_html=True)
            
            cutoff = st.slider("Punto de corte", 40, 60, 50)
            
            # Generar datos para RD
            x = np.arange(0, 101)
            base_y = 10 + 0.5 * x + np.random.normal(0, 5, len(x))
            
            # Control (antes del cutoff)
            y_control = np.where(x < cutoff, base_y, np.nan)
            # Tratamiento (después del cutoff, con salto)
            y_treatment = np.where(x >= cutoff, base_y + 15, np.nan)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x[~np.isnan(y_control)], 
                y=y_control[~np.isnan(y_control)], 
                mode='markers', 
                name='Control (No Tratamiento)',
                marker_color='#667eea'
            ))
            fig.add_trace(go.Scatter(
                x=x[~np.isnan(y_treatment)], 
                y=y_treatment[~np.isnan(y_treatment)], 
                mode='markers', 
                name='Tratamiento',
                marker_color='#fc8181'
            ))
            
            # Línea vertical en el cutoff
            fig.add_vline(x=cutoff, line_dash="dash", line_color="red", 
                         annotation_text=f"Corte: {cutoff}")
            
            fig.update_layout(
                title='Regresión por Discontinuidad',
                xaxis_title='Variable de Asignación',
                yaxis_title='Resultado'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Calcular el salto en la discontinuidad
            jump_size = 15  # El salto que programamos
            st.markdown(f'<div class="metric-card"><h2 style="color: #667eea;">{jump_size}</h2><p>Efecto Causal Estimado</p></div>', unsafe_allow_html=True)
            
            st.markdown('<div class="info-box">El "salto" en el punto de corte estima el efecto causal del tratamiento.</div>', unsafe_allow_html=True)
        
        with st.expander("⚖️ Variables Instrumentales"):
            st.markdown('<div class="info-box">Usa una variable que afecte el tratamiento pero no el resultado directamente, para estimar efectos causales.</div>', unsafe_allow_html=True)
            
            # Simulación de variables instrumentales
            st.markdown("#### Ejemplo: Efecto de la Educación en los Ingresos")
            st.markdown("**Instrumento:** Distancia a la Universidad")
            
            # Generar datos sintéticos
            np.random.seed(42)
            n = 200
            distance_to_uni = np.random.uniform(1, 50, n)  # Instrumento
            education = 12 + np.maximum(0, (20 - distance_to_uni) / 4) + np.random.normal(0, 2, n)  # Tratamiento
            income = 20000 + education * 3000 + np.random.normal(0, 5000, n)  # Resultado
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=distance_to_uni, y=education, mode='markers', 
                                        name='Datos', marker_color='#667eea'))
                # Línea de tendencia
                z = np.polyfit(distance_to_uni, education, 1)
                p = np.poly1d(z)
                fig1.add_trace(go.Scatter(x=sorted(distance_to_uni), y=p(sorted(distance_to_uni)), 
                                        mode='lines', name='Tendencia', line_color='red'))
                fig1.update_layout(title='Instrumento → Tratamiento', 
                                 xaxis_title='Distancia a Universidad (km)', 
                                 yaxis_title='Años de Educación')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=education, y=income, mode='markers', 
                                        name='Datos', marker_color='#48bb78'))
                # Línea de tendencia
                z2 = np.polyfit(education, income, 1)
                p2 = np.poly1d(z2)
                fig2.add_trace(go.Scatter(x=sorted(education), y=p2(sorted(education)), 
                                        mode='lines', name='Tendencia', line_color='red'))
                fig2.update_layout(title='Tratamiento → Resultado', 
                                 xaxis_title='Años de Educación', 
                                 yaxis_title='Ingresos Anuales ($)')
                st.plotly_chart(fig2, use_container_width=True)
            
            # Estimación IV vs OLS
            from scipy import stats
            
            # OLS (puede estar sesgado)
            slope_ols, intercept_ols, r_value_ols, p_value_ols, std_err_ols = stats.linregress(education, income)
            
            # IV (primera etapa)
            slope_first, intercept_first, _, _, _ = stats.linregress(distance_to_uni, education)
            
            # IV (forma reducida)
            slope_reduced, intercept_reduced, _, _, _ = stats.linregress(distance_to_uni, income)
            
            # Estimador IV
            iv_estimate = slope_reduced / slope_first
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<div class="metric-card"><h2 style="color: #667eea;">${slope_ols:.0f}</h2><p>Efecto OLS<br>(Potencialmente Sesgado)</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h2 style="color: #48bb78;">${iv_estimate:.0f}</h2><p>Efecto IV<br>(Causal)</p></div>', unsafe_allow_html=True)
        
        with st.expander("🎲 Experimentos Naturales"):
            st.markdown('<div class="info-box">Aprovecha eventos o políticas que asignan tratamientos de forma quasi-aleatoria.</div>', unsafe_allow_html=True)
            
            st.markdown("#### Ejemplo: Impacto de una Nueva Ley Antidiscriminación")
            
            # Datos simulados para diferencias en diferencias
            years = list(range(2015, 2025))
            
            # Grupo tratado (empresas en estado con nueva ley)
            treated_before = [20, 22, 21, 23, 24]  # 2015-2019 (antes de la ley)
            treated_after = [35, 38, 40, 42, 45]   # 2020-2024 (después de la ley)
            
            # Grupo control (empresas en estado sin nueva ley)
            control_before = [18, 19, 20, 21, 22]  # 2015-2019
            control_after = [23, 24, 25, 26, 27]   # 2020-2024
            
            fig = go.Figure()
            
            # Líneas pre-tratamiento
            fig.add_trace(go.Scatter(x=years[:5], y=treated_before, mode='lines+markers', 
                                   name='Tratado (Estado con Ley)', line_color='#48bb78'))
            fig.add_trace(go.Scatter(x=years[:5], y=control_before, mode='lines+markers', 
                                   name='Control (Estado sin Ley)', line_color='#667eea'))
            
            # Líneas post-tratamiento
            fig.add_trace(go.Scatter(x=years[5:], y=treated_after, mode='lines+markers', 
                                   name='Tratado (Post-Ley)', line_color='#48bb78', 
                                   line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=years[5:], y=control_after, mode='lines+markers', 
                                   name='Control (Post)', line_color='#667eea',
                                   line=dict(dash='dash')))
            
            # Línea vertical marcando la intervención
            fig.add_vline(x=2019.5, line_dash="dash", line_color="red", 
                         annotation_text="Nueva Ley")
            
            fig.update_layout(
                title='Experimento Natural: Diferencias en Diferencias',
                xaxis_title='Año',
                yaxis_title='% de Diversidad en Alta Gerencia'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Calcular efecto diferencias en diferencias
            treated_change = np.mean(treated_after) - np.mean(treated_before)
            control_change = np.mean(control_after) - np.mean(control_before)
            did_effect = treated_change - control_change
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-card"><h2 style="color: #48bb78;">+{treated_change:.1f}%</h2><p>Cambio en Tratado</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h2 style="color: #667eea;">+{control_change:.1f}%</h2><p>Cambio en Control</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><h2 style="color: #fc8181;">+{did_effect:.1f}%</h2><p>Efecto DiD</p></div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="success-box">✅ <strong>Efecto Causal Estimado:</strong> La nueva ley aumentó la diversidad en {did_effect:.1f} puntos porcentuales.</div>', unsafe_allow_html=True)

# Información adicional en el sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Recursos")
st.sidebar.markdown("""
- [Documentación de Fairness](https://fairlearn.org/)
- [Guías de Implementación](https://github.com/fairlearn/fairlearn)
- [Casos de Estudio](https://www.microsoft.com/en-us/research/theme/fairness-accountability-transparency-and-ethics-in-ai/)
""")

st.sidebar.markdown("### 💡 Consejos")
st.sidebar.info("""
**Para mejores resultados:**
1. Identifica el tipo de sesgo primero
2. Considera las restricciones de tu proyecto
3. Mide el impacto antes y después
4. Documenta todas las decisiones
5. Valida con expertos del dominio
""")

st.sidebar.markdown("### 📊 Métricas de Equidad")
with st.sidebar.expander("Definiciones"):
    st.markdown("""
    **Paridad Demográfica:** Misma tasa de decisiones positivas por grupo
    
    **Igualdad de Oportunidades:** Misma TPR (sensibilidad) por grupo
    
    **Odds Ecualizadas:** Misma TPR y FPR por grupo
    
    **Calibración:** Misma precisión de probabilidades por grupo
    """)

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; padding: 20px;">'
    '🎯 Plataforma de Equidad en IA - Versión Streamlit Completa<br>'
    '<small>Desarrollado para promover la IA responsable y equitativa</small>'
    '</div>', 
    unsafe_allow_html=True
)