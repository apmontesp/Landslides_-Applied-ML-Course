"""
Dashboard interactivo — Detección de Deslizamientos con Machine Learning en Colombia
Visualización de Datos · Entregable Final · Plotly + Streamlit
"""

import os, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deslizamientos ML · Colombia",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main { background:#FFFFFF; }
  .block-container { padding-top:1.4rem; padding-bottom:2rem; }
  h1 { color:#111827; }
  h2 { color:#1F2937; border-bottom:2px solid #DC2626; padding-bottom:5px; }
  .arg-box { background:#FEF2F2; border-left:4px solid #DC2626;
             padding:11px 15px; border-radius:4px; margin-bottom:.9rem;
             font-size:.94rem; color:#7F1D1D; }
  .hall-box { background:#F9FAFB; border-left:4px solid #6B7280;
              padding:9px 13px; border-radius:4px; margin-bottom:.7rem;
              font-size:.9rem; color:#374151; }
  .kpi-card { background:#F9FAFB; border:1px solid #E5E7EB;
              border-radius:8px; padding:14px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# DATOS
# ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, '..', 'data')

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'comparison_table.csv'))
    for col in ['F1 medio','Std','AUC-ROC','Precisión','Recall','IoU']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    FOLDS_FB = {
        'LR':             [0.7929,0.7681,0.8232,0.7772,0.7813],
        'SVM':            [0.7526,0.7731,0.8100,0.8325,0.8189],
        'RF':             [0.8363,0.8238,0.8480,0.8401,0.8361],
        'ResNet-50':      [0.7762,0.7708,0.8154,0.7636,0.8065],
        'U-Net ResNet34': [0.6855,0.7084,0.7061,0.6900,0.6842],
    }
    try:
        def lf(fname, key='best_f1', alt='f1_pixel_thr05'):
            with open(os.path.join(DATA_DIR,'folds',fname)) as f:
                d = json.load(f)
            return [x.get(key, x.get(alt, 0)) for x in d['folds']]
        folds = {
            'LR':             lf('logistic_regression_folds.json'),
            'SVM':            lf('svm_folds.json'),
            'RF':             lf('random_forest_folds.json'),
            'ResNet-50':      lf('resnet50_folds.json', key='f1_thr05'),
            'U-Net ResNet34': lf('unet_folds.json',    key='f1_pixel_thr05'),
        }
    except Exception:
        folds = FOLDS_FB
    return df, folds

df, fold_data = load_data()

# ──────────────────────────────────────────────────────────────────
# PALETA Y CONSTANTES
# ──────────────────────────────────────────────────────────────────
C = {
    'red':'#DC2626','gray':'#9CA3AF','dark':'#374151',
    'clasico':'#6B7280','dl':'#C4B5FD',
    'RedEdge':'#DC2626','Topo':'#F97316','SAR':'#EAB308','Optico':'#3B82F6',
    'blue_dark':'#1E3A5F','blue_light':'#93C5FD',
    'bg':'white','grid':'#EEEEEE',
}

TODOS = ['LR','SVM','RF','ResNet-50','EfficientNet','U-Net']
F1_VALS = {'LR':0.7886,'SVM':0.7974,'RF':0.8368,'ResNet-50':0.7840,'EfficientNet':0.7554,'U-Net':0.4443}
TIPO    = {'LR':'Clásico','SVM':'Clásico','RF':'Clásico','ResNet-50':'DL','EfficientNet':'DL','U-Net':'DL'}

def col_modelo(m):
    if m == 'RF': return C['red']
    return C['clasico'] if TIPO.get(m)=='Clásico' else C['gray']

def layout_base(title='', xlab='', ylab='', height=420):
    return dict(
        title=dict(text=title, font=dict(size=14, color='#111827'), x=0.02),
        xaxis=dict(title=xlab, gridcolor=C['grid'], showline=True,
                   linecolor='#CCCCCC', zeroline=False),
        yaxis=dict(title=ylab, gridcolor=C['grid'], showline=False,
                   tickfont=dict(size=11)),
        plot_bgcolor=C['bg'], paper_bgcolor=C['bg'],
        height=height, margin=dict(l=20,r=20,t=50,b=40),
        legend=dict(bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#E5E7EB', borderwidth=1,
                    font=dict(size=10)),
        hoverlabel=dict(bgcolor='white', font_size=12,
                        bordercolor='#E5E7EB'),
    )

# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏔️ Landslide ML · Colombia")
    st.markdown("---")
    seccion = st.radio("Navegación",
        ["Inicio","Exploración","Argumento","Conclusión"], index=0)
    st.markdown("---")
    st.markdown("**Filtros globales**")
    umbral = st.slider("Umbral F1 de referencia", 0.60, 0.95, 0.80, 0.01)
    sel = st.multiselect("Modelos a mostrar", TODOS, default=TODOS)
    mostrar_lit = st.toggle("Mostrar benchmarks literatura", value=True)
    st.markdown("---")
    st.caption("Datos: Landslide4Sense Dataset\nModelos: Google Colab")

# ──────────────────────────────────────────────────────────────────
# GRÁFICAS — funciones Plotly
# ──────────────────────────────────────────────────────────────────

def fig_f1_barras(modelos, umbral_f1):
    orden = sorted(modelos, key=lambda m: F1_VALS.get(m, 0))
    colors = [col_modelo(m) for m in orden]
    f1s = [F1_VALS[m] for m in orden]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=orden, x=f1s,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.3f}' for v in f1s],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>F1-Score: %{x:.4f}<br>Tipo: ' +
                      '<extra></extra>',
        customdata=[[TIPO.get(m,'—'), 'Modelo con mejor resultado' if m=='RF' else ''] for m in orden],
        hovertemplate='<b>%{y}</b><br>F1-Score: %{x:.4f}<br>Tipo: %{customdata[0]}<extra></extra>',
    ))
    fig.add_vline(x=umbral_f1, line_dash='dash', line_color=C['dark'], line_width=1.5,
                  annotation_text=f'F1={umbral_f1:.2f}',
                  annotation_position='top right',
                  annotation_font_color=C['dark'])
    fig.update_layout(**layout_base(
        title='F1-Score por Modelo — Detección de Deslizamientos',
        xlab='F1-Score (0 = no detecta nada · 1 = perfecto)',
        ylab='Modelo', height=max(350, len(orden)*60)
    ))
    fig.update_xaxes(range=[0, 0.97])
    return fig


def fig_pr_scatter(modelos_vis, mostrar_lit, umbral_f1):
    datos_pr = {
        'LR':(0.7971,0.7806,'Clásico'),
        'SVM':(0.8193,0.7777,'Clásico'),
        'RF':(0.7439,0.9569,'Clásico'),
        'ResNet-50':(0.7219,0.8771,'DL'),
    }
    lit = [
        ('Ghorbanzadeh et al. (2022)',  0.717,'#FECACA'),
        ('Lv et al. — L4S (2022)',      0.739,'#F87171'),
        ('Liu et al. — Multi-scale (2024)', 0.760,'#EF4444'),
        ('Enhanced U-Net++ (2025)',     0.841,'#B91C1C'),
    ]
    fig = go.Figure()

    if mostrar_lit:
        r_arr = np.linspace(0.63, 0.999, 300)
        for nombre, f1v, cl in lit:
            p_arr = f1v * r_arr / (2*r_arr - f1v)
            mask = (p_arr > 0) & (p_arr <= 1.0)
            fig.add_trace(go.Scatter(
                x=r_arr[mask], y=p_arr[mask], mode='lines',
                line=dict(color=cl, dash='dash', width=1.3),
                name=nombre, legendgroup='lit',
                hovertemplate=f'<b>{nombre}</b><br>F1 reportado: {f1v}<br>Recall: %{{x:.3f}}<br>Precisión: %{{y:.3f}}<extra></extra>',
            ))

    for nm, (prec, rec, tipo) in datos_pr.items():
        if nm not in modelos_vis:
            continue
        mk = 'diamond' if tipo == 'DL' else 'circle'
        fig.add_trace(go.Scatter(
            x=[rec], y=[prec], mode='markers+text',
            name=nm,
            marker=dict(size=14, color=col_modelo(nm),
                        line=dict(color='white', width=2),
                        symbol=mk),
            text=[nm], textposition='top right',
            textfont=dict(size=11, color=col_modelo(nm)),
            hovertemplate=f'<b>{nm}</b><br>Recall: {rec:.4f}<br>Precisión: {prec:.4f}<br>Tipo: {tipo}<extra></extra>',
        ))

    fig.update_layout(**layout_base(
        title='Precisión vs Cobertura por Modelo',
        xlab='Cobertura (Recall) — fracción de deslizamientos reales detectados',
        ylab='Precisión — de las alertas, ¿cuántas son reales?',
        height=500,
    ))
    fig.update_xaxes(range=[0.62, 1.02])
    fig.update_yaxes(range=[0.62, 0.90])
    return fig


def fig_canales():
    canales = [
        ('S2-B7 RedEdge3',0.8073,'RedEdge','Sentinel-2 B7'),
        ('S2-B6 RedEdge2',0.5625,'RedEdge','Sentinel-2 B6'),
        ('ALOS DEM',      0.1954,'Topo',   'Modelo de elevación digital'),
        ('S1-VH SAR',     0.1882,'SAR',    'Sentinel-1 polarización VH'),
        ('DEM Slope',     0.0430,'Topo',   'Pendiente derivada del DEM'),
        ('S2-B8A NIR-A',  0.0221,'Optico', 'Sentinel-2 B8A NIR estrecho'),
    ]
    fig = go.Figure()
    for nm, delta, grp, desc in canales:
        fig.add_trace(go.Bar(
            y=[nm], x=[delta],
            orientation='h',
            name=grp,
            legendgroup=grp,
            showlegend=(nm == [c[0] for c in canales if c[2]==grp][0]),
            marker_color=C[grp],
            hovertemplate=f'<b>{nm}</b><br>Δ brecha de señal: {delta:.4f}<br>Sensor: {desc}<br>Grupo: {grp}<extra></extra>',
        ))
    fig.update_layout(**layout_base(
        title='Canales Satelitales más Discriminativos',
        xlab='Brecha de señal (Δ) entre zonas con y sin deslizamiento',
        ylab='Canal satelital', height=380,
    ))
    fig.update_xaxes(range=[0, 0.97])
    return fig


def fig_dot_plot():
    dot_data = [
        ('S2-B7 RedEdge3',2.0209,1.2136,'RedEdge'),
        ('S2-B6 RedEdge2',1.4782,0.9157,'RedEdge'),
        ('ALOS DEM',      1.2739,1.0786,'Topo'),
        ('S1-VH SAR',     1.2488,1.0606,'SAR'),
        ('S2-B8A NIR-A',  1.0397,1.0176,'Optico'),
        ('DEM Slope',     1.0703,1.0274,'Topo'),
    ]
    fig = go.Figure()
    for nm, pos, neg, grp in dot_data:
        # Línea conectora
        fig.add_trace(go.Scatter(
            x=[neg, pos], y=[nm, nm], mode='lines',
            line=dict(color='#D1D5DB', width=2),
            showlegend=False,
            hoverinfo='skip',
        ))
        # Punto sólido = con deslizamiento
        fig.add_trace(go.Scatter(
            x=[pos], y=[nm], mode='markers',
            name=f'{grp} — con deslizamiento',
            legendgroup=grp,
            showlegend=(nm == [d[0] for d in dot_data if d[3]==grp][0]),
            marker=dict(size=12, color=C[grp],
                        line=dict(color='white', width=2)),
            hovertemplate=f'<b>{nm}</b><br>Con deslizamiento: {pos:.4f}<br>Sin deslizamiento: {neg:.4f}<br>Δ = {pos-neg:.4f}<br>Grupo: {grp}<extra></extra>',
        ))
        # Punto hueco = sin deslizamiento
        fig.add_trace(go.Scatter(
            x=[neg], y=[nm], mode='markers',
            showlegend=False,
            marker=dict(size=12, color='white',
                        line=dict(color=C[grp], width=2.5)),
            hovertemplate=f'<b>{nm}</b><br>Sin deslizamiento: {neg:.4f}<br>Con deslizamiento: {pos:.4f}<br>Δ = {pos-neg:.4f}<extra></extra>',
        ))

    fig.update_layout(**layout_base(
        title='Brecha de Señal entre Clases — Top 6 Canales<br><sup>Sólido = con deslizamiento · Hueco = sin deslizamiento · Δ = brecha</sup>',
        xlab='Reflectancia media normalizada',
        ylab='Canal satelital', height=400,
    ))
    fig.update_xaxes(range=[0.8, 2.55])
    return fig


def fig_folds(folds_dict, umbral_f1):
    orden = sorted(folds_dict.keys(), key=lambda k: np.mean(folds_dict[k]), reverse=True)
    fig = go.Figure()
    for m in orden:
        vals = folds_dict[m]
        media = np.mean(vals)
        std   = np.std(vals, ddof=1)
        color = C['red'] if 'RF' in m else '#9CA3AF'
        fig.add_trace(go.Box(
            x=vals, y=[m]*len(vals),
            orientation='h',
            name=m,
            boxpoints='all',
            jitter=0.4,
            pointpos=0,
            marker=dict(size=9, color=color,
                        line=dict(color='white', width=1.5)),
            line=dict(color=color),
            fillcolor=color.replace(')', ',0.35)').replace('rgb','rgba') if 'rgb' in color else color + '55',
            hovertemplate=f'<b>{m}</b><br>F1 fold: %{{x:.4f}}<br>Media: {media:.4f}<br>Std: {std:.4f}<extra></extra>',
        ))
    fig.add_vline(x=umbral_f1, line_dash='dash', line_color=C['dark'], line_width=1.5,
                  annotation_text=f'F1={umbral_f1:.2f}',
                  annotation_position='bottom right',
                  annotation_font_color=C['dark'])
    fig.update_layout(**layout_base(
        title='Consistencia del modelo como indicador de confianza',
        xlab='F1-Score por fold — cada punto es un experimento independiente',
        ylab='Modelo', height=max(380, len(orden)*70),
    ))
    fig.update_traces(boxmean=True)
    return fig


def fig_protocolos(modelos_cl, modelos_dl, umbral_f1):
    f1_opt  = {'LR':0.7886,'SVM':0.7974,'RF':0.8368}
    f1_base = {'LR':0.7512,'SVM':0.7340,'RF':0.7891}
    f1_dl   = {'ResNet-50':0.7840,'EfficientNet':0.7554,'U-Net':0.4443}

    fig = go.Figure()
    # Barras 14 bandas
    if modelos_cl:
        fig.add_trace(go.Bar(
            name='14 bandas del satélite',
            x=modelos_cl,
            y=[f1_opt[m] for m in modelos_cl],
            marker_color=C['blue_dark'],
            text=[f'{f1_opt[m]:.3f}' for m in modelos_cl],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>14 bandas: %{y:.4f}<br>Protocolo optimizado (n=1500)<extra></extra>',
        ))
        fig.add_trace(go.Bar(
            name='Características básicas del terreno',
            x=modelos_cl,
            y=[f1_base[m] for m in modelos_cl],
            marker_color=C['blue_light'],
            text=[f'{f1_base[m]:.3f}' for m in modelos_cl],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Básico: %{y:.4f}<br>Comparable literatura (n=3799)<extra></extra>',
        ))
    if modelos_dl:
        fig.add_trace(go.Bar(
            name='Deep Learning',
            x=modelos_dl,
            y=[f1_dl[m] for m in modelos_dl],
            marker_color='#E5E7EB',
            text=[f'{f1_dl[m]:.3f}' for m in modelos_dl],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>F1: %{y:.4f}<br>Deep Learning<extra></extra>',
        ))
    fig.add_hline(y=umbral_f1, line_dash='dash', line_color=C['dark'], line_width=1.5,
                  annotation_text=f'F1={umbral_f1:.2f}',
                  annotation_position='top right')
    fig.update_layout(**layout_base(
        title='El protocolo de evaluación cambia el resultado',
        xlab='Modelo', ylab='F1-Score', height=460,
    ))
    fig.update_layout(barmode='group', yaxis_range=[0.3, 0.97])
    return fig


def fig_complejidad(modelos, umbral_f1):
    comp = {'LR':'Baja','SVM':'Media','RF':'Media','ResNet-50':'Alta','EfficientNet':'Alta','U-Net':'Alta'}
    orden = sorted(modelos, key=lambda m: F1_VALS.get(m,0))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=orden,
        x=[F1_VALS[m] for m in orden],
        orientation='h',
        marker_color=[col_modelo(m) for m in orden],
        text=[f'{F1_VALS[m]:.3f}' for m in orden],
        textposition='outside',
        customdata=[[TIPO.get(m,'—'), comp.get(m,'—')] for m in orden],
        hovertemplate='<b>%{y}</b><br>F1-Score: %{x:.4f}<br>Tipo: %{customdata[0]}<br>Complejidad: %{customdata[1]}<extra></extra>',
    ))
    fig.add_vline(x=umbral_f1, line_dash='dash', line_color=C['dark'], line_width=1.5,
                  annotation_text=f'F1={umbral_f1:.2f}',
                  annotation_position='top right',
                  annotation_font_color=C['dark'])
    fig.update_layout(**layout_base(
        title='Complejidad del modelo vs. resultado real<br><sup>Más parámetros no garantizan mejor detección</sup>',
        xlab='F1-Score — capacidad de detección',
        ylab='Modelo', height=max(350, len(orden)*62),
    ))
    fig.update_xaxes(range=[0, 0.97])
    return fig


def fig_canales_colombia():
    canales = [
        ('S2-B7 RedEdge3',0.8073,'RedEdge',True,'Sentinel-2 B7 — mayor discriminación'),
        ('S2-B6 RedEdge2',0.5625,'RedEdge',True,'Sentinel-2 B6'),
        ('ALOS DEM',      0.1954,'Topo',   False,'Modelo de elevación ALOS'),
        ('S1-VH SAR',     0.1882,'SAR',    True,'Sentinel-1 polarización VH'),
        ('DEM Slope',     0.0430,'Topo',   False,'Pendiente derivada del DEM'),
        ('S2-B8A NIR-A',  0.0221,'Optico', True,'Sentinel-2 B8A NIR estrecho'),
    ]
    fig = go.Figure()
    for nm, delta, grp, disp, desc in canales:
        estado = 'Copernicus (gratuito)' if disp else 'Requiere DEM externo'
        fig.add_trace(go.Bar(
            y=[nm], x=[delta],
            orientation='h',
            name=grp,
            legendgroup=grp,
            showlegend=(nm == [c[0] for c in canales if c[2]==grp][0]),
            marker_color=C[grp],
            hovertemplate=f'<b>{nm}</b><br>Δ: {delta:.4f}<br>Sensor: {desc}<br>Disponibilidad: {estado}<extra></extra>',
        ))
    fig.update_layout(**layout_base(
        title='Canales más discriminativos y su disponibilidad para Colombia',
        xlab='Brecha de señal (Δ)',
        ylab='Canal satelital', height=380,
    ))
    fig.update_xaxes(range=[0, 0.97])
    return fig


def fig_consistencia(folds_dict, umbral_f1):
    orden = sorted(folds_dict.keys(), key=lambda k: np.mean(folds_dict[k]), reverse=True)
    fig = go.Figure()
    for m in orden:
        vals = folds_dict[m]
        media = np.mean(vals); std = np.std(vals,ddof=1)
        color = C['red'] if 'RF' in m else '#9CA3AF'
        for i, v in enumerate(vals):
            fig.add_trace(go.Scatter(
                x=[v], y=[m],
                mode='markers',
                name=m,
                legendgroup=m,
                showlegend=(i == 0),
                marker=dict(size=11, color=color,
                            line=dict(color='white', width=1.5)),
                hovertemplate=f'<b>{m}</b><br>Fold {i+1}: {v:.4f}<br>Media: {media:.4f}  Std: {std:.4f}<extra></extra>',
            ))
        # Línea media
        fig.add_trace(go.Scatter(
            x=[media-std, media+std], y=[m, m],
            mode='lines',
            line=dict(color=color, width=8),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip',
        ))
    fig.add_vline(x=umbral_f1, line_dash='dash', line_color=C['dark'], line_width=1.5,
                  annotation_text=f'F1={umbral_f1:.2f}',
                  annotation_position='bottom right',
                  annotation_font_color=C['dark'])
    fig.update_layout(**layout_base(
        title='Consistencia del modelo como indicador de confianza<br><sup>En Colombia, los datos serán escasos — la variabilidad importa más</sup>',
        xlab='F1-Score por fold — cada punto es un experimento independiente',
        ylab='Modelo', height=max(380, len(orden)*72),
    ))
    return fig

# ──────────────────────────────────────────────────────────────────
# SECCIÓN: INICIO
# ──────────────────────────────────────────────────────────────────
if seccion == "Inicio":
    st.markdown("# Detección de Deslizamientos con Machine Learning")
    st.markdown("### ¿Qué nos dicen los modelos sobre cómo proteger Colombia?")
    st.markdown("---")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<div class="kpi-card"><div style="font-size:2.2rem;font-weight:700;color:#DC2626">400–600</div><div>eventos/año en Colombia<br><small style="color:#9CA3AF">SGC, 2023</small></div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi-card"><div style="font-size:2.2rem;font-weight:700;color:#DC2626">0</div><div>datasets etiquetados colombianos disponibles públicamente</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi-card"><div style="font-size:2.2rem;font-weight:700;color:#DC2626">14</div><div>bandas satelitales analizadas<br>Sentinel-1/2 + DEM</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="arg-box"><strong>Argumento central:</strong><br>"El modelo más sofisticado no siempre gana — y en Colombia, donde no hay un dataset propio, elegir mal el modelo y el protocolo de evaluación puede ser la diferencia entre una herramienta útil y una que falla cuando más se necesita."</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Recorrido del análisis")
        pasos = [("1. Exploración","¿Qué muestran los datos sin hipótesis?"),
                 ("2. Trampa metodológica","El resultado cambia según cómo evalúes"),
                 ("3. Colombia vs mundo","¿Dónde estamos frente a la literatura?"),
                 ("4. Complejidad","Más parámetros ≠ mejor detección"),
                 ("5. Disponibilidad","¿Qué información satelital necesita Colombia?"),
                 ("6. Confianza","¿En qué modelo confiar con datos escasos?")]
        for paso,desc in pasos:
            st.markdown(f'<div class="hall-box"><strong>{paso}</strong> — {desc}</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown("#### Modelos evaluados")
        resumen = pd.DataFrame({
            'Modelo': TODOS, 'Tipo': [TIPO[m] for m in TODOS],
            'F1-Score': [F1_VALS[m] for m in TODOS],
        }).sort_values('F1-Score',ascending=False).reset_index(drop=True)
        st.dataframe(resumen, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────
# SECCIÓN: EXPLORACIÓN
# ──────────────────────────────────────────────────────────────────
elif seccion == "Exploración":
    st.markdown("# Fase Exploratoria — ¿Dónde estamos?")
    st.caption("Pasa el cursor sobre las gráficas para ver los valores exactos · Haz clic en la leyenda para mostrar/ocultar series · Arrastra para hacer zoom")
    st.markdown("---")

    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "H1 · F1 por modelo","H2 · Precisión vs Cobertura",
        "H3 · Canales satelitales","H4 · Brecha de señal","H5 · Variabilidad folds",
    ])
    modelos_vis = [m for m in TODOS if m in sel]

    with tab1:
        st.markdown("**¿Qué tan bien detecta cada modelo un deslizamiento?**")
        if modelos_vis:
            st.plotly_chart(fig_f1_barras(modelos_vis, umbral), use_container_width=True)
        else:
            st.warning("Selecciona al menos un modelo.")

    with tab2:
        st.markdown("**¿Precisión o cobertura? ¿Podemos tener los dos?**")
        st.plotly_chart(fig_pr_scatter(modelos_vis, mostrar_lit, umbral), use_container_width=True)

    with tab3:
        st.markdown("**¿Qué información satelital separa mejor las clases?**")
        st.plotly_chart(fig_canales(), use_container_width=True)

    with tab4:
        st.markdown("**¿Cómo se diferencian las clases en los canales clave?**")
        st.plotly_chart(fig_dot_plot(), use_container_width=True)

    with tab5:
        st.markdown("**¿El rendimiento es estable entre experimentos?**")
        folds_vis = {k:v for k,v in fold_data.items() if any(k.startswith(m.split()[0]) for m in sel)} or fold_data
        st.plotly_chart(fig_folds(folds_vis, umbral), use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# SECCIÓN: ARGUMENTO
# ──────────────────────────────────────────────────────────────────
elif seccion == "Argumento":
    st.markdown("# Fase Aclaratoria — Del hallazgo al argumento")
    st.caption("Pasa el cursor sobre los elementos para ver valores · Clic en leyenda para aislar series")
    st.markdown('<div class="arg-box">Hilo conductor: <strong>Problema Colombia → ¿qué modelo responde?</strong></div>', unsafe_allow_html=True)
    st.markdown("---")

    taba1,taba2,taba3,taba4,taba5 = st.tabs([
        "A1 · La trampa metodológica","A2 · Colombia vs mundo",
        "A3 · Complejidad ≠ rendimiento","A4 · Qué necesita Colombia","A5 · Confianza",
    ])
    modelos_vis = [m for m in TODOS if m in sel]
    m_cl = [m for m in ['LR','SVM','RF'] if m in sel]
    m_dl = [m for m in ['ResNet-50','EfficientNet','U-Net'] if m in sel]

    with taba1:
        cc1,cc2 = st.columns([1,2])
        with cc1:
            st.markdown("### El resultado cambia según cómo evalúes el modelo")
            st.markdown('<div class="arg-box">Si se reporta solo el protocolo más favorable, se puede <strong>sobrestimar el rendimiento</strong> en condiciones locales.</div>', unsafe_allow_html=True)
            st.markdown("**14 bandas:** protocolo optimizado, n=1500")
            st.markdown("**Básicas:** comparable a literatura, n=3799")
        with cc2:
            st.plotly_chart(fig_protocolos(m_cl, m_dl, umbral), use_container_width=True)

    with taba2:
        cc1,cc2 = st.columns([1,2])
        with cc1:
            st.markdown("### ¿Dónde estamos frente a la literatura internacional?")
            st.markdown('<div class="arg-box">Los modelos de referencia se entrenaron en Nepal, Perú e Italia. Colombia tiene <strong>terreno andino con régimen de lluvias diferente</strong> — y ningún dataset etiquetado.</div>', unsafe_allow_html=True)
        with cc2:
            st.plotly_chart(fig_pr_scatter(modelos_vis, mostrar_lit, umbral), use_container_width=True)

    with taba3:
        cc1,cc2 = st.columns([1,2])
        with cc1:
            st.markdown("### Complejidad del modelo vs. resultado real")
            st.markdown('<div class="arg-box"><strong>La objeción más común:</strong> "Los modelos profundos siempre son mejores."<br>U-Net — la más compleja — tiene el <strong>peor F1</strong>.</div>', unsafe_allow_html=True)
        with cc2:
            st.plotly_chart(fig_complejidad(modelos_vis, umbral), use_container_width=True)

    with taba4:
        cc1,cc2 = st.columns([1,2])
        with cc1:
            st.markdown("### ¿Qué información satelital necesita Colombia?")
            st.markdown('<div class="arg-box">Las bandas <strong>RedEdge</strong> (Sentinel-2) son las más discriminativas y están disponibles gratuitamente.<br>El desafío: <strong>etiquetas post-evento</strong>.</div>', unsafe_allow_html=True)
        with cc2:
            st.plotly_chart(fig_canales_colombia(), use_container_width=True)

    with taba5:
        cc1,cc2 = st.columns([1,2])
        with cc1:
            st.markdown("### ¿En cuál modelo confiar con datos escasos?")
            st.markdown('<div class="arg-box">La <strong>consistencia entre folds</strong> es el indicador más importante cuando los datos son pocos y de un solo evento.</div>', unsafe_allow_html=True)
            st.markdown("**RF:** Std=0.008 — el más consistente")
            st.markdown("**SVM:** Std=0.033 — mayor variabilidad")
        with cc2:
            folds_a5 = {k:v for k,v in fold_data.items() if any(k.startswith(m.split()[0]) for m in sel)} or fold_data
            st.plotly_chart(fig_consistencia(folds_a5, umbral), use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# SECCIÓN: CONCLUSIÓN
# ──────────────────────────────────────────────────────────────────
elif seccion == "Conclusión":
    st.markdown("# Conclusión — Lo que necesita Colombia")
    st.markdown('<div class="arg-box" style="font-size:1.05rem">La arquitectura viene después del contexto. Random Forest — interpretable y eficiente con pocos datos — es un punto de partida más sólido que redes profundas diseñadas para miles de imágenes segmentadas.</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("### Síntesis de hallazgos")
        tabla = pd.DataFrame({
            'Hallazgo': ['F1 por modelo','Precisión vs Cobertura','Canales satelitales','Brecha de señal','Variabilidad folds'],
            'Observación': ['RF supera F1=0.80; U-Net queda muy por debajo','RF prioriza cobertura (Recall=0.96)',
                            'Bandas RedEdge dominan la discriminación','RedEdge3 tiene brecha 4× mayor que SAR-VH','RF es el más consistente (Std=0.008)'],
        })
        st.dataframe(tabla, use_container_width=True, hide_index=True)
        st.markdown("### Tabla comparativa")
        df_s = df.copy()
        for c in ['F1 medio','Precisión','Recall']:
            df_s[c] = df_s[c].map(lambda x: f'{x:.4f}' if pd.notna(x) else '—')
        st.dataframe(df_s[['Modelo','Tipo','F1 medio','Precisión','Recall']], use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Condiciones para Colombia")
        for cond,estado,bg,bcolor,impl in [
            ("Dataset etiquetado nacional","No existe","#FEE2E2","#DC2626","Sin esto, cualquier modelo es extrapolación"),
            ("Bandas satelitales","Sentinel-2 disponible (RedEdge)","#DCFCE7","#16A34A","La señal está — faltan etiquetas post-evento"),
            ("Protocolo de evaluación","Depende del estudio","#FEF9C3","#CA8A04","Usar 5 folds, comparable a literatura"),
            ("Arquitectura apropiada","RF como punto de partida","#DCFCE7","#16A34A","Interpretable, eficiente con pocos datos"),
        ]:
            st.markdown(f'<div style="background:{bg};border-left:4px solid {bcolor};padding:10px 14px;border-radius:4px;margin-bottom:9px"><strong style="color:{bcolor}">{cond}</strong><br><span style="color:#374151">{estado}</span><br><small style="color:#6B7280">{impl}</small></div>', unsafe_allow_html=True)

        st.markdown("### Próximos pasos")
        st.markdown("""
1. **Etiquetar** imágenes históricas del SGC (Servicio Geológico Colombiano)
2. **Descargar** bandas RedEdge de Sentinel-2 via API Copernicus
3. **Entrenar y evaluar** con 5 folds y protocolo comparable a literatura
        """)
    st.markdown("---")
    st.caption("Datos: Landslide4Sense · Modelos: Google Colab · Benchmarks: Ghorbanzadeh (2022), L4S (2022), Liu et al. (2024), Enhanced U-Net++ (2025)")
