"""
╔══════════════════════════════════════════════════════════════════════╗
║   LANDSLIDE ML — Dashboard Analítico de Detección de Deslizamientos  ║
║   Visualización de Datos | Asignatura: Visualización de Datos        ║
║   Ana Patricia Montes — 2026                                         ║
╚══════════════════════════════════════════════════════════════════════╝

Ejecutar:
    cd visualizacion_datos/03_dashboard
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from PIL import Image
import numpy as np

# ─── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Landslide ML · Dashboard Analítico",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paleta de colores ───────────────────────────────────────────────────────
COLORS = {
    "primary":    "#D62728",   # rojo alarma (landslide)
    "secondary":  "#1F77B4",   # azul datos
    "accent":     "#FF7F0E",   # naranja énfasis
    "neutral":    "#7F7F7F",   # gris base
    "bg_dark":    "#0E1117",
    "bg_card":    "#1A1D27",
    "clasico":    "#2CA02C",   # verde → modelos clásicos
    "dl":         "#9467BD",   # morado → deep learning
    "best":       "#D62728",   # rojo → mejor modelo
    "success":    "#27AE60",
    "warning":    "#E67E22",
}

MODEL_COLORS = {
    "Random Forest":     COLORS["clasico"],
    "SVM (RBF)":         "#4DAF4A",
    "Logistic Regression": "#A1D99B",
    "ResNet-50":         COLORS["dl"],
    "EfficientNet-B4":   "#C49BD3",
    "U-Net ResNet-34":   "#D0A9E0",
}

# ─── CSS personalizado ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* Fondo y fuentes generales */
[data-testid="stAppViewContainer"] {
    background: #0E1117;
}
[data-testid="stSidebar"] {
    background: #131722;
    border-right: 1px solid #2D3047;
}
.main-header {
    background: linear-gradient(135deg, #1A1D27 0%, #0E1117 100%);
    border: 1px solid #D62728;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
}
.main-header h1 { color: #FFFFFF; font-size: 1.8rem; margin: 0; }
.main-header p  { color: #9CA3AF; font-size: 0.9rem; margin: 4px 0 0; }
.kpi-card {
    background: #1A1D27;
    border: 1px solid #2D3047;
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
}
.kpi-value { font-size: 2rem; font-weight: 700; color: #D62728; }
.kpi-label { font-size: 0.8rem; color: #9CA3AF; text-transform: uppercase;
             letter-spacing: 0.08em; margin-top: 4px; }
.kpi-sub   { font-size: 0.75rem; color: #6B7280; margin-top: 2px; }
.finding-box {
    background: linear-gradient(135deg, #1A1D27, #121520);
    border-left: 4px solid #D62728;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 12px 0;
}
.finding-box strong { color: #D62728; }
.finding-box p { color: #D1D5DB; margin: 0; line-height: 1.6; }
.section-title {
    font-size: 1.1rem; font-weight: 600; color: #F3F4F6;
    border-bottom: 2px solid #D62728;
    padding-bottom: 6px; margin-bottom: 16px;
}
.insight-pill {
    display: inline-block;
    background: #D627281A;
    color: #F87171;
    border: 1px solid #D6272840;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    margin: 3px 2px;
}
</style>
""", unsafe_allow_html=True)

# ─── Rutas de datos ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
FIG_DIR  = os.path.join(DATA_DIR, "figures")

@st.cache_data
def load_comparison():
    return pd.read_csv(os.path.join(DATA_DIR, "comparison_table.csv"))

@st.cache_data
def load_summary():
    with open(os.path.join(DATA_DIR, "final_summary.json")) as f:
        return json.load(f)

@st.cache_data
def load_channels():
    return pd.read_csv(os.path.join(DATA_DIR, "channel_stats_by_class.csv"))

@st.cache_data
def load_class_stats():
    with open(os.path.join(DATA_DIR, "class_stats.json")) as f:
        return json.load(f)

def load_fig(name):
    path = os.path.join(FIG_DIR, name)
    if os.path.exists(path):
        return Image.open(path)
    return None

df_models   = load_comparison()
summary     = load_summary()
df_channels = load_channels()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏔️ Landslide ML")
    st.markdown("<p style='color:#9CA3AF;font-size:0.82rem;'>Dashboard Analítico — Visualización de Datos 2026</p>", unsafe_allow_html=True)
    st.divider()

    # Filtro tipo de modelo
    tipos_disponibles = ["Clásico", "Deep Learning"]
    tipos_sel = st.multiselect(
        "Tipo de Modelo",
        tipos_disponibles,
        default=tipos_disponibles,
        help="Filtra qué familias de modelos visualizar"
    )

    # Filtro métrica principal
    metrica = st.selectbox(
        "Métrica principal",
        ["F1 medio", "Precisión", "Recall", "IoU", "AUC-ROC"],
        index=0,
    )

    st.divider()
    st.markdown("<p style='color:#6B7280;font-size:0.75rem;'>Fuente: Landslide4Sense · Sentinel-1/2 · ALOS DEM<br>Dataset: 3799 imágenes · 14 canales · 5-fold CV</p>",
                unsafe_allow_html=True)

# ─── Filtrar datos según sidebar ─────────────────────────────────────────────
df_filt = df_models[df_models["Tipo"].isin(tipos_sel)].copy()

# ─── Header principal ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏔️ Detección de Deslizamientos con ML</h1>
  <p>Dashboard Analítico · Landslide4Sense Dataset · 14 canales multiespectrales · 6 modelos evaluados</p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs principales ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Comparación de Modelos",
    "🔬 Señales del Terreno",
    "🧠 Análisis Avanzado",
    "🌎 Transferibilidad",
    "🏆 Síntesis Final",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Comparación de Modelos
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Pregunta de negocio ──────────────────────────────────────────────────
    st.markdown("""
    <div class="finding-box">
      <p>❓ <strong>Pregunta de Negocio:</strong>
      ¿Qué modelo de Machine Learning maximiza la detección de deslizamientos de tierra
      en imágenes satelitales multiespectrales, balanceando precisión, cobertura (recall) y
      capacidad de generalización?</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    best = df_models.loc[df_models["F1 medio"].idxmax()]

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value">🥇 RF</div>
          <div class="kpi-label">Mejor Modelo</div>
          <div class="kpi-sub">Random Forest</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:#27AE60">0.837</div>
          <div class="kpi-label">F1 Score (best)</div>
          <div class="kpi-sub">±0.008 std</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:#3B82F6">95.7%</div>
          <div class="kpi-label">Recall</div>
          <div class="kpi-sub">Landslides detectados</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:#F59E0B">0.810</div>
          <div class="kpi-label">AUC-ROC</div>
          <div class="kpi-sub">Discriminación</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-value" style="color:#8B5CF6">6</div>
          <div class="kpi-label">Modelos Evaluados</div>
          <div class="kpi-sub">3 clásicos · 3 DL</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráficas comparativas ────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-title">Ranking por F1 Score (media ± std · 5-fold CV)</div>', unsafe_allow_html=True)

        df_plot = df_filt.sort_values(metrica if metrica in df_filt.columns else "F1 medio", ascending=True)
        col_metric = metrica if metrica in df_filt.columns else "F1 medio"

        colors_bar = [MODEL_COLORS.get(m, COLORS["neutral"]) for m in df_plot["Modelo"]]
        # Destacar mejor modelo
        colors_bar = [COLORS["best"] if m == "Random Forest" else c
                      for m, c in zip(df_plot["Modelo"], colors_bar)]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=df_plot["Modelo"],
            x=df_plot["F1 medio"],
            orientation="h",
            marker_color=colors_bar,
            error_x=dict(
                type="data",
                array=df_plot["Std"].fillna(0),
                color="#6B7280",
                thickness=2,
                width=6,
            ),
            text=[f"{v:.3f}" for v in df_plot["F1 medio"]],
            textposition="outside",
            textfont=dict(color="#F3F4F6", size=13),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "F1 medio: %{x:.3f}<br>"
                "<extra></extra>"
            ),
        ))

        # Línea de referencia
        fig_bar.add_vline(x=0.8, line_dash="dot", line_color="#6B7280",
                          annotation_text="F1 = 0.80", annotation_font_color="#9CA3AF",
                          annotation_position="top right")

        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F3F4F6",
            xaxis=dict(range=[0, 1.05], gridcolor="#2D3047", gridwidth=1,
                       title="F1 Score", title_font_size=12),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            height=340,
            margin=dict(l=10, r=30, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Hallazgo anotado
        st.markdown("""
        <div class="finding-box">
          <p>🔍 <strong>Hallazgo:</strong> Random Forest supera a las redes neuronales profundas en F1 Score
          (0.837 vs. 0.784 ResNet-50), con la menor varianza entre folds (±0.008).
          U-Net ResNet-34 —diseñada para segmentación— obtiene el peor resultado (0.444),
          evidenciando que la arquitectura por sí sola no garantiza rendimiento sin suficientes datos de entrenamiento.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-title">Perfil Multi-Métrica (Radar)</div>', unsafe_allow_html=True)

        # Radar chart — sólo modelos con todas las métricas
        df_radar = df_models[df_models["Tipo"] == "Clásico"].copy()
        categories = ["F1 medio", "Precisión", "Recall", "IoU", "AUC-ROC"]

        fig_radar = go.Figure()
        radar_colors = {
            "Random Forest": COLORS["best"],
            "SVM (RBF)":     COLORS["secondary"],
            "Logistic Regression": COLORS["neutral"],
        }
        for _, row in df_radar.iterrows():
            vals = [row.get(c, 0) for c in categories]
            vals_closed = vals + [vals[0]]
            cats_closed = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed,
                theta=cats_closed,
                fill="toself",
                fillcolor=radar_colors.get(row["Modelo"], "#7F7F7F") + "33",
                line=dict(color=radar_colors.get(row["Modelo"], "#7F7F7F"), width=2),
                name=row["Modelo"],
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0.6, 1.0],
                                tickfont=dict(size=9, color="#9CA3AF"),
                                gridcolor="#2D3047"),
                angularaxis=dict(tickfont=dict(size=10, color="#D1D5DB"),
                                 gridcolor="#2D3047"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#F3F4F6",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                        font=dict(size=10)),
            height=340,
            margin=dict(l=20, r=20, t=10, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Tabla de resultados filtrada ─────────────────────────────────────────
    st.markdown('<div class="section-title">Tabla de Métricas Completa</div>', unsafe_allow_html=True)

    df_table = df_filt[["Modelo", "Tipo", "F1 medio", "Std", "Precisión", "Recall", "IoU", "AUC-ROC"]].copy()
    df_table = df_table.sort_values("F1 medio", ascending=False).reset_index(drop=True)

    # Scatter Precisión vs Recall
    st.markdown('<div class="section-title">Trade-off: Precisión vs. Recall</div>', unsafe_allow_html=True)

    df_scatter = df_filt.dropna(subset=["Precisión", "Recall"]).copy()
    if not df_scatter.empty:
        fig_scatter = px.scatter(
            df_scatter,
            x="Precisión", y="Recall",
            color="Tipo",
            size="F1 medio",
            size_max=40,
            text="Modelo",
            color_discrete_map={"Clásico": COLORS["clasico"], "Deep Learning": COLORS["dl"]},
            hover_data={"F1 medio": ":.3f", "Tipo": True},
        )
        fig_scatter.update_traces(textposition="top center", textfont_size=11)
        # Curvas iso-F1
        for f1_target in [0.75, 0.80, 0.85]:
            prec = np.linspace(0.01, 1.0, 200)
            rec  = f1_target * prec / (2 * prec - f1_target + 1e-9)
            mask = (rec > 0) & (rec <= 1)
            fig_scatter.add_trace(go.Scatter(
                x=prec[mask], y=rec[mask],
                mode="lines", line=dict(dash="dot", color="#4B5563", width=1),
                name=f"F1={f1_target}", showlegend=(f1_target == 0.75),
                hoverinfo="skip",
            ))

        fig_scatter.add_annotation(x=0.744, y=0.957, text="RF: alto recall,<br>baja precisión",
                                   font=dict(color="#D62728", size=10),
                                   arrowcolor="#D62728", ax=40, ay=-40)

        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F3F4F6",
            xaxis=dict(gridcolor="#2D3047", title="Precisión", range=[0.65, 0.90]),
            yaxis=dict(gridcolor="#2D3047", title="Recall",    range=[0.70, 1.00]),
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("""
        <div class="finding-box">
          <p>🔍 <strong>Insight crítico:</strong> Random Forest maximiza el <strong>Recall (95.7%)</strong>
          —detecta casi todos los deslizamientos— a costa de menor Precisión (74.4%).
          Para alertas tempranas de desastres, priorizar Recall es la decisión correcta:
          un falso negativo (deslizamiento no detectado) es mucho más costoso que un falso positivo.</p>
        </div>
        """, unsafe_allow_html=True)

    # Tabla final
    st.dataframe(
        df_table.style
            .format({"F1 medio": "{:.3f}", "Std": "{:.3f}", "Precisión": "{:.3f}",
                     "Recall": "{:.3f}", "IoU": "{:.3f}", "AUC-ROC": lambda x: f"{x:.3f}" if x != "—" else "—"})
            .background_gradient(subset=["F1 medio"], cmap="RdYlGn", vmin=0.4, vmax=0.9),
        use_container_width=True,
        hide_index=True,
    )

    # ── Benchmarking con Literatura ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📚 Benchmarking con Estado del Arte</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="finding-box">
      <p>🔍 <strong>Contexto:</strong> ¿Cómo se posicionan nuestros modelos respecto a los trabajos
      publicados en detección de deslizamientos con Landslide4Sense?</p>
    </div>
    """, unsafe_allow_html=True)

    bench_tabs = st.tabs(["Comparación Global", "F1 vs AUC-ROC", "Variabilidad por Fold", "Significancia Estadística"])

    with bench_tabs[0]:
        col_a, col_b = st.columns(2)
        with col_a:
            img = load_fig("comparacion_literatura.png")
            if img:
                st.image(img, caption="Nuestros modelos vs. literatura publicada (Landslide4Sense benchmark)",
                         use_container_width=True)
        with col_b:
            img = load_fig("benchmarking_literatura.png")
            if img:
                st.image(img, caption="Benchmarking detallado contra métodos del estado del arte",
                         use_container_width=True)

    with bench_tabs[1]:
        img = load_fig("barras_f1_auc.png")
        if img:
            st.image(img, caption="F1 Score y AUC-ROC comparados entre todos los modelos evaluados",
                     use_container_width=True)

    with bench_tabs[2]:
        col_a, col_b = st.columns(2)
        with col_a:
            img = load_fig("boxplot_folds.png")
            if img:
                st.image(img, caption="Distribución de F1 Score por fold — variabilidad entre modelos",
                         use_container_width=True)
        with col_b:
            img = load_fig("L4S_01_classicos_5fold_comparison.png")
            if img:
                st.image(img, caption="Comparación clásicos 5-fold — consistencia entre folds",
                         use_container_width=True)

    with bench_tabs[3]:
        img = load_fig("wilcoxon_heatmap.png")
        if img:
            st.image(img, caption="Test de Wilcoxon: significancia estadística de las diferencias entre modelos (p-valor)",
                     use_container_width=True)
        st.markdown("""
        <div class="finding-box">
          <p>🔍 El heatmap de Wilcoxon confirma que la diferencia entre Random Forest y los modelos DL
          es <strong>estadísticamente significativa</strong> (p &lt; 0.05), no un artefacto del muestreo.</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Señales del Terreno (EDA)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div class="finding-box">
      <p>❓ <strong>Pregunta de Negocio:</strong>
      ¿Qué bandas espectrales y sensores (Sentinel-2 óptico, Sentinel-1 SAR, DEM de elevación)
      contienen la mayor señal discriminativa para diferenciar zonas de deslizamiento de zonas estables?</p>
    </div>
    """, unsafe_allow_html=True)

    # Slider de top canales
    top_n = st.slider("Número de canales a visualizar", min_value=5, max_value=14, value=10, step=1)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-title">Poder Discriminativo por Canal (|Δ media|)</div>', unsafe_allow_html=True)

        df_ch = df_channels.copy()
        df_ch["|Delta|"] = df_ch["Delta"].abs()
        df_ch = df_ch.sort_values("|Delta|", ascending=True).tail(top_n)

        # Categorías de sensor
        def sensor_cat(name):
            if "SAR" in name or "VV" in name or "VH" in name:
                return "SAR (Sentinel-1)"
            elif "DEM" in name or "Slope" in name:
                return "Topografía (DEM)"
            elif "RedEdge" in name:
                return "RedEdge (Sentinel-2)"
            else:
                return "Óptico (Sentinel-2)"

        df_ch["Sensor"] = df_ch["Nombre"].apply(sensor_cat)
        sensor_color = {
            "SAR (Sentinel-1)":    "#F59E0B",
            "Topografía (DEM)":    "#EF4444",
            "RedEdge (Sentinel-2)": "#D62728",
            "Óptico (Sentinel-2)":  "#3B82F6",
        }
        colors_ch = [sensor_color[s] for s in df_ch["Sensor"]]

        fig_ch = go.Figure()
        fig_ch.add_trace(go.Bar(
            y=df_ch["Nombre"],
            x=df_ch["|Delta|"],
            orientation="h",
            marker_color=colors_ch,
            text=[f"|Δ|={v:.3f}" for v in df_ch["|Delta|"]],
            textposition="outside",
            textfont=dict(color="#F3F4F6", size=11),
            hovertemplate="<b>%{y}</b><br>|Δ| = %{x:.4f}<extra></extra>",
        ))

        # Anotación en el canal más discriminativo
        max_ch = df_ch.loc[df_ch["|Delta|"].idxmax()]
        fig_ch.add_annotation(
            y=max_ch["Nombre"],
            x=max_ch["|Delta|"] + 0.02,
            text="⭐ Más discriminativo",
            font=dict(color="#D62728", size=10),
            showarrow=False,
            xanchor="left",
        )

        fig_ch.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F3F4F6",
            xaxis=dict(gridcolor="#2D3047", title="|Δ media| (Positivo − Negativo)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            height=400,
            margin=dict(l=10, r=80, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_ch, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Media por Clase (Landslide vs. No-Landslide)</div>', unsafe_allow_html=True)

        df_top5 = df_channels.sort_values("Delta", key=abs, ascending=False).head(6)

        fig_cmp = go.Figure()
        x = list(range(len(df_top5)))
        fig_cmp.add_trace(go.Bar(
            x=df_top5["Nombre"], y=df_top5["Media_Pos"],
            name="Landslide ✓",
            marker_color=COLORS["primary"],
            error_y=dict(type="data", array=df_top5["Std_Pos"], color="#9CA3AF"),
        ))
        fig_cmp.add_trace(go.Bar(
            x=df_top5["Nombre"], y=df_top5["Media_Neg"],
            name="No-Landslide ✗",
            marker_color=COLORS["secondary"],
            error_y=dict(type="data", array=df_top5["Std_Neg"], color="#9CA3AF"),
        ))

        fig_cmp.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F3F4F6",
            xaxis=dict(gridcolor="#2D3047", tickangle=-30),
            yaxis=dict(gridcolor="#2D3047", title="Media normalizada"),
            legend=dict(orientation="h", yanchor="bottom", y=1.0),
            height=400,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    # Hallazgo EDA
    st.markdown("""
    <div class="finding-box">
      <p>🔍 <strong>Hallazgo EDA:</strong>
      Los canales <strong>RedEdge3 (B7, Δ=0.807)</strong> y <strong>RedEdge2 (B6, Δ=0.563)</strong>
      de Sentinel-2 son los más discriminativos: zonas de deslizamiento presentan valores
      sistemáticamente más altos en estas bandas, posiblemente por la exposición de suelo desnudo
      y material removido. El <strong>DEM de elevación (Δ=0.195)</strong> y <strong>SAR-VH (Δ=0.188)</strong>
      confirman el rol clave de la topografía y la rugosidad superficial.</p>
    </div>
    """, unsafe_allow_html=True)

    # Imagen ejemplo (si existe)
    st.markdown('<div class="section-title">Ejemplo Visual — Dataset Landslide4Sense</div>', unsafe_allow_html=True)
    img_samples = load_fig("fig1_samples_pos_neg.png")
    if img_samples:
        st.image(img_samples, caption="Muestras del dataset: Positivos (deslizamiento) vs. Negativos (terreno estable)",
                 use_container_width=True)

    # Correlación canales
    img_corr = load_fig("fig5_correlation_matrix.png")
    if img_corr:
        st.markdown('<div class="section-title">Matriz de Correlación entre Canales</div>', unsafe_allow_html=True)
        st.image(img_corr, caption="Correlación de Pearson entre los 14 canales de entrada",
                 use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Análisis Avanzado
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    st.markdown("""
    <div class="finding-box">
      <p>❓ <strong>Pregunta de Negocio:</strong>
      ¿Por qué el modelo toma sus decisiones? ¿Qué regiones de la imagen activan las predicciones
      de landslide y cuán calibradas están las probabilidades predichas?</p>
    </div>
    """, unsafe_allow_html=True)

    analysis_option = st.radio(
        "Selecciona el análisis:",
        ["SHAP — Importancia de Features", "GradCAM — Mapas de Activación",
         "Calibración y Confianza", "Análisis de Incertidumbre",
         "Curvas de Entrenamiento", "Visualización de Predicciones"],
        horizontal=True,
    )

    if analysis_option == "SHAP — Importancia de Features":
        col_a, col_b = st.columns(2)
        with col_a:
            img = load_fig("shap_importancia_features.png")
            if img:
                st.image(img, caption="SHAP: Importancia global de features (Random Forest)",
                         use_container_width=True)
        with col_b:
            img = load_fig("shap_por_sensor.png")
            if img:
                st.image(img, caption="SHAP: Contribución agregada por tipo de sensor",
                         use_container_width=True)
        st.markdown("""
        <div class="finding-box">
          <p>🔍 Los valores SHAP confirman que los canales <strong>RedEdge y DEM</strong> dominan
          la explicabilidad del modelo. La elevación y la pendiente son los predictores físicos
          más importantes: los deslizamientos ocurren preferentemente en pendientes pronunciadas.</p>
        </div>
        """, unsafe_allow_html=True)

    elif analysis_option == "GradCAM — Mapas de Activación":
        tabs_gc = st.tabs(["ResNet-50", "EfficientNet-B4", "Comparativa", "Solo SAR"])
        figs_gc = [
            ("gradcam_resnet50.png", "GradCAM — ResNet-50: activaciones en zonas de deslizamiento"),
            ("gradcam_efficientnet_b4.png", "GradCAM — EfficientNet-B4: foco en bordes de ruptura"),
            ("gradcam_comparison.png", "Comparativa GradCAM: ResNet-50 vs. EfficientNet-B4"),
            ("gradcam_sar_only.png", "GradCAM — Sólo canales SAR (Sentinel-1 VV/VH)"),
        ]
        for tab_gc, (fname, caption) in zip(tabs_gc, figs_gc):
            with tab_gc:
                img = load_fig(fname)
                if img:
                    st.image(img, caption=caption, use_container_width=True)

    elif analysis_option == "Calibración y Confianza":
        col_a, col_b = st.columns(2)
        with col_a:
            img = load_fig("curva_calibracion.png")
            if img:
                st.image(img, caption="Curva de Calibración: probabilidades predichas vs. frecuencia real",
                         use_container_width=True)
        with col_b:
            img = load_fig("varianza_vs_error.png")
            if img:
                st.image(img, caption="Varianza de predicción vs. Error del modelo",
                         use_container_width=True)
        st.markdown("""
        <div class="finding-box">
          <p>🔍 El modelo Random Forest muestra <strong>buena calibración</strong> en el rango [0.3–0.7],
          con ligera sobreconfianza en predicciones extremas. Para sistemas de alerta temprana,
          se recomienda aplicar calibración isotónica antes de desplegar.</p>
        </div>
        """, unsafe_allow_html=True)

    elif analysis_option == "Análisis de Incertidumbre":
        col_a, col_b = st.columns(2)
        with col_a:
            img = load_fig("distribucion_incertidumbre.png")
            if img:
                st.image(img, caption="Distribución de incertidumbre predictiva por clase",
                         use_container_width=True)
        with col_b:
            img = load_fig("mapa_features_fisicos.png")
            if img:
                st.image(img, caption="Mapa de features físicos del terreno",
                         use_container_width=True)

    elif analysis_option == "Curvas de Entrenamiento":
        st.markdown('<div class="section-title">Convergencia durante el entrenamiento (5-fold CV)</div>',
                    unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            img = load_fig("training_curves_5fold.png")
            if img:
                st.image(img, caption="Curvas de entrenamiento 5-fold — todos los modelos DL",
                         use_container_width=True)
        with col_b:
            img = load_fig("dl_best_epochs.png")
            if img:
                st.image(img, caption="Mejor época por fold — criterio de early stopping",
                         use_container_width=True)
        col_c, col_d = st.columns(2)
        with col_c:
            img = load_fig("unet_curvas_entrenamiento.png")
            if img:
                st.image(img, caption="U-Net ResNet-34 — curvas de loss y F1 por fold",
                         use_container_width=True)
        with col_d:
            img = load_fig("unet_analisis_overfitting.png")
            if img:
                st.image(img, caption="U-Net — análisis de overfitting: gap train/val",
                         use_container_width=True)
        st.markdown("""
        <div class="finding-box">
          <p>🔍 Las curvas de entrenamiento revelan que <strong>U-Net sufre overfitting severo</strong>
          a partir del fold 3, con un gap train/val creciente que explica su bajo F1=0.444.
          ResNet-50 y EfficientNet-B4 convergen establemente pero no alcanzan al Random Forest,
          que no requiere entrenamiento iterativo.</p>
        </div>
        """, unsafe_allow_html=True)

    else:  # Visualización de Predicciones
        st.markdown('<div class="section-title">Predicciones del modelo sobre imágenes de test</div>',
                    unsafe_allow_html=True)
        pred_tabs = st.tabs(["Predicciones Fold 1", "Mapa de Confianza", "Mapa de Errores"])
        with pred_tabs[0]:
            img = load_fig("predicciones_fold1.png")
            if img:
                st.image(img, caption="Predicciones vs. ground truth — muestra representativa del Fold 1",
                         use_container_width=True)
        with pred_tabs[1]:
            img = load_fig("confianza_fold1.png")
            if img:
                st.image(img, caption="Mapa de confianza: probabilidad predicha por píxel (Fold 1)",
                         use_container_width=True)
        with pred_tabs[2]:
            img = load_fig("error_map_todos_folds.png")
            if img:
                st.image(img, caption="Mapa de errores acumulado — falsos positivos y negativos en todos los folds",
                         use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Transferibilidad
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    st.markdown("""
    <div class="finding-box">
      <p>❓ <strong>Pregunta de Negocio:</strong>
      ¿Los modelos entrenados en el dataset global Landslide4Sense generalizan a regiones
      geográficas no vistas durante el entrenamiento, incluyendo el caso específico de Colombia?</p>
    </div>
    """, unsafe_allow_html=True)

    transfer_option = st.selectbox(
        "Experimento de transferibilidad:",
        ["LORO — Leave-One-Region-Out", "Colombia — Adaptación Regional",
         "Robustez ante Nubosidad", "Ablación de Sensores"],
    )

    if transfer_option == "LORO — Leave-One-Region-Out":
        st.markdown("""
        <div class="finding-box">
          <p>❓ <strong>Diseño experimental:</strong> En cada fold, se excluye completamente una región
          geográfica del entrenamiento y se evalúa sobre ella. Las 4 regiones son:
          <strong>Iburi (Japón), Kodagu (India), Gorkha (Nepal), Taiwan.</strong></p>
        </div>
        """, unsafe_allow_html=True)

        loro_tabs = st.tabs(["Resumen LORO", "Iburi 🇯🇵", "Kodagu 🇮🇳", "Gorkha 🇳🇵", "Taiwan 🇹🇼", "Convergencia"])

        with loro_tabs[0]:
            col_a, col_b = st.columns(2)
            with col_a:
                img = load_fig("loro_vs_5fold_comparison.png")
                if img:
                    st.image(img, caption="LORO vs. 5-fold estándar: impacto de la generalización geográfica",
                             use_container_width=True)
            with col_b:
                img = load_fig("precision_recall.png")
                if img:
                    st.image(img, caption="Precisión-Recall por región LORO",
                             use_container_width=True)

        regiones = [
            ("region0_iburi_predictions.png",   "Iburi, Japón — predicciones vs. ground truth"),
            ("region1_kodagu_predictions.png",  "Kodagu, India — predicciones vs. ground truth"),
            ("region2_gorkha_predictions.png",  "Gorkha, Nepal — predicciones vs. ground truth"),
            ("region3_taiwan_predictions.png",  "Taiwan — predicciones vs. ground truth"),
        ]
        for tab_r, (fname, caption) in zip(loro_tabs[1:5], regiones):
            with tab_r:
                img = load_fig(fname)
                if img:
                    st.image(img, caption=caption, use_container_width=True)
                else:
                    st.info(f"Figura no disponible: {fname}")

        with loro_tabs[5]:
            img = load_fig("loro_convergence_curves.png")
            if img:
                st.image(img, caption="Curvas de convergencia por región LORO",
                         use_container_width=True)

        st.markdown("""
        <div class="finding-box">
          <p>🔍 <strong>Hallazgo LORO:</strong> El modelo generaliza con F1 > 0.75 en las 4 regiones,
          pero con variabilidad notable. <strong>Gorkha (Nepal)</strong> presenta el mayor desafío
          por la topografía andina compleja, mientras que <strong>Iburi (Japón)</strong> muestra
          la mejor transferencia por características espectrales más uniformes.</p>
        </div>
        """, unsafe_allow_html=True)

    elif transfer_option == "Colombia — Adaptación Regional":
        col_a, col_b = st.columns(2)
        with col_a:
            img = load_fig("criticidad_sensores_colombia.png")
            if img:
                st.image(img, caption="Criticidad de sensores en Colombia: SAR-VH gana importancia",
                         use_container_width=True)
        with col_b:
            img = load_fig("curva_aprendizaje.png")
            if img:
                st.image(img, caption="Curva de aprendizaje con datos de Colombia: ¿cuántas muestras se necesitan?",
                         use_container_width=True)
        img_col = load_fig("transferibilidad_colombia.png")
        if img_col:
            st.image(img_col, caption="Síntesis: transferibilidad del modelo hacia Colombia",
                     use_container_width=True)
        st.markdown("""
        <div class="finding-box">
          <p>🔍 La criticidad de los sensores varía en Colombia: el DEM pierde importancia
          relativa en topografía andina compleja, mientras que <strong>SAR-VH gana protagonismo</strong>
          por su robustez ante la nubosidad persistente de la región.</p>
        </div>
        """, unsafe_allow_html=True)

    elif transfer_option == "Robustez ante Nubosidad":
        col_a, col_b = st.columns(2)
        with col_a:
            img = load_fig("robustez_nubosidad.png")
            if img:
                st.image(img, caption="Degradación de F1 según % de cobertura nubosa",
                         use_container_width=True)
        with col_b:
            img = load_fig("ensamble_modelos.png")
            if img:
                st.image(img, caption="Ensamble RF+SAR: mayor robustez ante nubosidad",
                         use_container_width=True)

    else:  # Ablación de Sensores
        col_a, col_b = st.columns(2)
        with col_a:
            img = load_fig("ablacion_sensores.png")
            if img:
                st.image(img, caption="Ablación: impacto en F1 al eliminar cada tipo de sensor",
                         use_container_width=True)
        with col_b:
            img = load_fig("ablacion_rf_vs_unet.png")
            if img:
                st.image(img, caption="Ablación comparativa: Random Forest vs. U-Net por sensor",
                         use_container_width=True)
        img_sz = load_fig("analisis_tamano.png")
        if img_sz:
            st.image(img_sz, caption="Análisis de tamaño de muestra: ¿cuántos datos necesita cada modelo?",
                     use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Síntesis Final
# ══════════════════════════════════════════════════════════════════════════════
with tab5:

    st.markdown("""
    <div class="finding-box">
      <p>❓ <strong>Pregunta de Negocio:</strong>
      ¿Dónde está posicionado este proyecto respecto al estado del arte? ¿Qué brechas quedan
      por cerrar y cuál es el camino hacia un sistema de detección de deslizamientos de producción?</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs de cierre
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="kpi-card">
          <div class="kpi-value" style="color:#27AE60">0.837</div>
          <div class="kpi-label">Mejor F1</div><div class="kpi-sub">Random Forest</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="kpi-card">
          <div class="kpi-value" style="color:#3B82F6">6</div>
          <div class="kpi-label">Modelos Evaluados</div><div class="kpi-sub">3 clásicos · 3 DL</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="kpi-card">
          <div class="kpi-value" style="color:#F59E0B">4</div>
          <div class="kpi-label">Regiones LORO</div><div class="kpi-sub">Iburi·Kodagu·Gorkha·Taiwan</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="kpi-card">
          <div class="kpi-value" style="color:#8B5CF6">14</div>
          <div class="kpi-label">Canales</div><div class="kpi-sub">Óptico·SAR·DEM</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Evolución del proyecto
    st.markdown('<div class="section-title">Evolución del Proyecto — Fase 1 → Fase 2</div>',
                unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        img = load_fig("evolucion_fase1_fase2.png")
        if img:
            st.image(img, caption="Evolución de métricas: comparación Fase 1 (2-fold) vs. Fase 2 (5-fold comparable)",
                     use_container_width=True)
    with col_b:
        img = load_fig("evolucion_fases.png")
        if img:
            st.image(img, caption="Evolución completa del proyecto por fases",
                     use_container_width=True)

    # Gap Analysis
    st.markdown('<div class="section-title">Gap Analysis — Proyecto vs. Estado del Arte</div>',
                unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        img = load_fig("gap_analysis_radar.png")
        if img:
            st.image(img, caption="Radar de brecha: nuestro proyecto vs. mejores modelos publicados",
                     use_container_width=True)
    with col_b:
        img = load_fig("literatura_vs_proyecto.png")
        if img:
            st.image(img, caption="Posicionamiento del proyecto en el espacio de resultados publicados",
                     use_container_width=True)

    # Síntesis visual global
    st.markdown('<div class="section-title">Síntesis Global y Roadmap</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        img = load_fig("gap_analysis.png")
        if img:
            st.image(img, caption="Análisis de brecha: dimensiones pendientes de mejora",
                     use_container_width=True)
    with col_b:
        img = load_fig("roadmap.png")
        if img:
            st.image(img, caption="Roadmap: próximos pasos hacia producción",
                     use_container_width=True)

    st.markdown("""
    <div class="finding-box">
      <p>🏆 <strong>Conclusión del proyecto:</strong>
      Random Forest con 14 canales multiespectrales alcanza F1=0.837, posicionándose
      <strong>competitivamente respecto al estado del arte</strong> en Landslide4Sense.
      La brecha principal respecto a los top papers se encuentra en la segmentación pixel-a-pixel
      (IoU), donde arquitecturas especializadas con más datos de entrenamiento superan al enfoque clásico.
      El experimento LORO confirma generalización razonable a regiones no vistas,
      con Colombia como caso de uso prioritario para trabajo futuro.</p>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#6B7280;font-size:0.8rem;'>"
    "Landslide ML · Dashboard Analítico · Visualización de Datos 2026 · "
    "Datos: Landslide4Sense (Sentinel-1/2, ALOS DEM) · "
    "Modelos: Random Forest, SVM, Logistic Regression, ResNet-50, EfficientNet-B4, U-Net ResNet-34"
    "</p>",
    unsafe_allow_html=True,
)
