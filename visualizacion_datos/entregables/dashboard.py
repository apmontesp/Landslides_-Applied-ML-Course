"""
Dashboard interactivo — Deteccion de Deslizamientos con Machine Learning en Colombia
Visualizacion de Datos - Entregable Final - Plotly + Streamlit
"""

import os, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Deslizamientos ML - Colombia",
    page_icon="mountain",
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

# ── DATOS ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, '..', 'data')

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'comparison_table.csv'))
    for col in ['F1 medio', 'Std', 'AUC-ROC', 'Precision', 'Recall', 'IoU']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    FOLDS_FB = {
        'LR':             [0.7929, 0.7681, 0.8232, 0.7772, 0.7813],
        'SVM':            [0.7526, 0.7731, 0.8100, 0.8325, 0.8189],
        'RF':             [0.8363, 0.8238, 0.8480, 0.8401, 0.8361],
        'ResNet-50':      [0.7762, 0.7708, 0.8154, 0.7636, 0.8065],
        'U-Net ResNet34': [0.6855, 0.7084, 0.7061, 0.6900, 0.6842],
    }
    try:
        def lf(fname, key='best_f1', alt='f1_pixel_thr05'):
            with open(os.path.join(DATA_DIR, 'folds', fname)) as fh:
                d = json.load(fh)
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

# ── PALETA ────────────────────────────────────────────────────────
C = {
    'red': '#DC2626', 'gray': '#9CA3AF', 'dark': '#374151',
    'clasico': '#6B7280', 'dl': '#C4B5FD',
    'RedEdge': '#DC2626', 'Topo': '#F97316', 'SAR': '#EAB308', 'Optico': '#3B82F6',
    'blue_dark': '#1E3A5F', 'blue_light': '#93C5FD',
    'bg': 'white', 'grid': '#EEEEEE',
}

TODOS   = ['LR', 'SVM', 'RF', 'ResNet-50', 'EfficientNet', 'U-Net']
F1_VALS = {'LR': 0.7886, 'SVM': 0.7974, 'RF': 0.8368,
           'ResNet-50': 0.7840, 'EfficientNet': 0.7554, 'U-Net': 0.4443}
TIPO    = {'LR': 'Clasico', 'SVM': 'Clasico', 'RF': 'Clasico',
           'ResNet-50': 'DL', 'EfficientNet': 'DL', 'U-Net': 'DL'}

def col_modelo(m):
    if m == 'RF':
        return C['red']
    return C['clasico'] if TIPO.get(m) == 'Clasico' else C['gray']

def layout_base(title='', xlab='', ylab='', height=420):
    return dict(
        title=dict(text=title, font=dict(size=14, color='#111827'), x=0.02),
        xaxis=dict(title=xlab, gridcolor=C['grid'], showline=True,
                   linecolor='#CCCCCC', zeroline=False),
        yaxis=dict(title=ylab, gridcolor=C['grid'], showline=False,
                   tickfont=dict(size=11)),
        plot_bgcolor=C['bg'], paper_bgcolor=C['bg'],
        height=height, margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#E5E7EB', borderwidth=1, font=dict(size=10)),
        hoverlabel=dict(bgcolor='white', font_size=12, bordercolor='#E5E7EB'),
    )

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Landslide ML - Colombia")
    st.markdown("---")
    seccion = st.radio("Navegacion",
                       ["Inicio", "Exploracion", "Argumento", "Conclusion"],
                       index=0)
    st.markdown("---")
    st.markdown("**Filtros globales**")
    umbral = st.slider("Umbral F1 de referencia", 0.60, 0.95, 0.80, 0.01)
    sel = st.multiselect("Modelos a mostrar", TODOS, default=TODOS)
    mostrar_lit = st.toggle("Mostrar benchmarks literatura", value=True)
    st.markdown("---")
    st.caption("Datos: Landslide4Sense Dataset | Modelos: Google Colab")

# ── FUNCIONES PLOTLY ─────────────────────────────────────────────

def fig_f1_barras(modelos, umbral_f1):
    orden  = sorted(modelos, key=lambda m: F1_VALS.get(m, 0))
    colors = [col_modelo(m) for m in orden]
    f1s    = [F1_VALS[m] for m in orden]
    tipos  = [TIPO.get(m, '—') for m in orden]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=orden, x=f1s, orientation='h',
        marker_color=colors,
        text=["{:.3f}".format(v) for v in f1s],
        textposition='outside',
        customdata=tipos,
        hovertemplate='<b>%{y}</b><br>F1-Score: %{x:.4f}<br>Tipo: %{customdata}<extra></extra>',
    ))
    fig.add_vline(x=umbral_f1, line_dash='dash', line_color=C['dark'], line_width=1.5,
                  annotation_text='F1={:.2f}'.format(umbral_f1),
                  annotation_position='top right',
                  annotation_font_color=C['dark'])
    fig.update_layout(**layout_base(
        title='F1-Score por Modelo — Deteccion de Deslizamientos',
        xlab='F1-Score (0 = no detecta nada - 1 = perfecto)',
        ylab='Modelo', height=max(350, len(orden) * 60)
    ))
    fig.update_xaxes(range=[0, 0.97])
    return fig


def fig_pr_scatter(modelos_vis, mostrar_lit_flag, umbral_f1):
    datos_pr = {
        'LR':        (0.7971, 0.7806, 'Clasico'),
        'SVM':       (0.8193, 0.7777, 'Clasico'),
        'RF':        (0.7439, 0.9569, 'Clasico'),
        'ResNet-50': (0.7219, 0.8771, 'DL'),
    }
    lit = [
        ('Ghorbanzadeh et al. (2022)',       0.717, '#FECACA'),
        ('Lv et al. - L4S (2022)',           0.739, '#F87171'),
        ('Liu et al. - Multi-scale (2024)',  0.760, '#EF4444'),
        ('Enhanced U-Net++ (2025)',          0.841, '#B91C1C'),
    ]
    fig = go.Figure()
    if mostrar_lit_flag:
        r_arr = np.linspace(0.63, 0.999, 300)
        for nombre, f1v, cl in lit:
            p_arr = f1v * r_arr / (2 * r_arr - f1v)
            mask  = (p_arr > 0) & (p_arr <= 1.0)
            ht = '<b>{}</b><br>F1 reportado: {}<br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>'.format(nombre, f1v)
            fig.add_trace(go.Scatter(
                x=r_arr[mask], y=p_arr[mask], mode='lines',
                line=dict(color=cl, dash='dash', width=1.3),
                name=nombre, legendgroup='lit',
                hovertemplate=ht,
            ))
    for nm, (prec, rec, tipo) in datos_pr.items():
        if nm not in modelos_vis:
            continue
        mk = 'diamond' if tipo == 'DL' else 'circle'
        ht = '<b>{}</b><br>Recall: {:.4f}<br>Precision: {:.4f}<br>Tipo: {}<extra></extra>'.format(nm, rec, prec, tipo)
        fig.add_trace(go.Scatter(
            x=[rec], y=[prec], mode='markers+text',
            name=nm,
            marker=dict(size=14, color=col_modelo(nm),
                        line=dict(color='white', width=2), symbol=mk),
            text=[nm], textposition='top right',
            textfont=dict(size=11, color=col_modelo(nm)),
            hovertemplate=ht,
        ))
    fig.update_layout(**layout_base(
        title='Precision vs Cobertura por Modelo',
        xlab='Cobertura (Recall) — fraccion de deslizamientos reales detectados',
        ylab='Precision — de las alertas, cuantas son reales?',
        height=500,
    ))
    fig.update_xaxes(range=[0.62, 1.02])
    fig.update_yaxes(range=[0.62, 0.90])
    return fig


def fig_canales():
    canales = [
        ('S2-B7 RedEdge3', 0.8073, 'RedEdge', 'Sentinel-2 B7'),
        ('S2-B6 RedEdge2', 0.5625, 'RedEdge', 'Sentinel-2 B6'),
        ('ALOS DEM',       0.1954, 'Topo',    'Modelo de elevacion digital'),
        ('S1-VH SAR',      0.1882, 'SAR',     'Sentinel-1 polarizacion VH'),
        ('DEM Slope',      0.0430, 'Topo',    'Pendiente derivada del DEM'),
        ('S2-B8A NIR-A',   0.0221, 'Optico',  'Sentinel-2 B8A NIR estrecho'),
    ]
    fig = go.Figure()
    grp_shown = set()
    for nm, delta, grp, desc in canales:
        show = grp not in grp_shown
        grp_shown.add(grp)
        ht = '<b>{}</b><br>Delta: {:.4f}<br>Sensor: {}<br>Grupo: {}<extra></extra>'.format(nm, delta, desc, grp)
        fig.add_trace(go.Bar(
            y=[nm], x=[delta], orientation='h',
            name=grp, legendgroup=grp, showlegend=show,
            marker_color=C[grp],
            hovertemplate=ht,
        ))
    fig.update_layout(**layout_base(
        title='Canales Satelitales mas Discriminativos',
        xlab='Brecha de senal (Delta) entre zonas con y sin deslizamiento',
        ylab='Canal satelital', height=380,
    ))
    fig.update_xaxes(range=[0, 0.97])
    return fig


def fig_dot_plot():
    dot_data = [
        ('S2-B7 RedEdge3', 2.0209, 1.2136, 'RedEdge'),
        ('S2-B6 RedEdge2', 1.4782, 0.9157, 'RedEdge'),
        ('ALOS DEM',       1.2739, 1.0786, 'Topo'),
        ('S1-VH SAR',      1.2488, 1.0606, 'SAR'),
        ('S2-B8A NIR-A',   1.0397, 1.0176, 'Optico'),
        ('DEM Slope',      1.0703, 1.0274, 'Topo'),
    ]
    fig = go.Figure()
    grp_shown = set()
    for nm, pos, neg, grp in dot_data:
        show = grp not in grp_shown
        grp_shown.add(grp)
        ht = '<b>{}</b><br>Con deslizamiento: {:.4f}<br>Sin deslizamiento: {:.4f}<br>Delta: {:.4f}<extra></extra>'.format(
            nm, pos, neg, pos - neg)
        fig.add_trace(go.Scatter(
            x=[neg, pos], y=[nm, nm], mode='lines',
            line=dict(color='#D1D5DB', width=2),
            showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=[pos], y=[nm], mode='markers',
            name=grp, legendgroup=grp, showlegend=show,
            marker=dict(size=12, color=C[grp], line=dict(color='white', width=2)),
            hovertemplate=ht,
        ))
        fig.add_trace(go.Scatter(
            x=[neg], y=[nm], mode='markers',
            showlegend=False,
            marker=dict(size=12, color='white', line=dict(color=C[grp], width=2.5)),
            hovertemplate=ht,
        ))
    fig.update_layout(**layout_base(
        title='Brecha de Senal entre Clases - Top 6 Canales',
        xlab='Reflectancia media normalizada',
        ylab='Canal satelital', height=400,
    ))
    fig.update_xaxes(range=[0.8, 2.55])
    return fig


def fig_folds(folds_dict, umbral_f1):
    orden = sorted(folds_dict.keys(), key=lambda k: np.mean(folds_dict[k]), reverse=True)
    fig = go.Figure()
    for m in orden:
        vals  = folds_dict[m]
        media = np.mean(vals)
        std   = np.std(vals, ddof=1)
        color = C['red'] if 'RF' in m else '#9CA3AF'
        ht = '<b>{}</b><br>F1 fold: %{{x:.4f}}<br>Media: {:.4f} | Std: {:.4f}<extra></extra>'.format(m, media, std)
        fig.add_trace(go.Box(
            x=vals, y=[m] * len(vals),
            orientation='h', name=m,
            boxpoints='all', jitter=0.4, pointpos=0,
            marker=dict(size=9, color=color, line=dict(color='white', width=1.5)),
            line=dict(color=color),
            fillcolor=color + '44',
            hovertemplate=ht,
        ))
    fig.add_vline(x=umbral_f1, line_dash='dash', line_color=C['dark'], line_width=1.5,
                  annotation_text='F1={:.2f}'.format(umbral_f1),
                  annotation_position='bottom right',
                  annotation_font_color=C['dark'])
    fig.update_layout(**layout_base(
        title='Consistencia del modelo como indicador de confianza',
        xlab='F1-Score por fold — cada punto es un experimento independiente',
        ylab='Modelo', height=max(380, len(orden) * 70),
    ))
    fig.update_traces(boxmean=True)
    return fig


def fig_protocolos(modelos_cl, modelos_dl, umbral_f1):
    f1_opt  = {'LR': 0.7886, 'SVM': 0.7974, 'RF': 0.8368}
    f1_base = {'LR': 0.7512, 'SVM': 0.7340, 'RF': 0.7891}
    f1_dl   = {'ResNet-50': 0.7840, 'EfficientNet': 0.7554, 'U-Net': 0.4443}
    fig = go.Figure()
    if modelos_cl:
        fig.add_trace(go.Bar(
            name='14 bandas del satelite',
            x=modelos_cl,
            y=[f1_opt[m] for m in modelos_cl],
            marker_color=C['blue_dark'],
            text=["{:.3f}".format(f1_opt[m]) for m in modelos_cl],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>14 bandas: %{y:.4f}<br>Protocolo optimizado (n=1500)<extra></extra>',
        ))
        fig.add_trace(go.Bar(
            name='Caracteristicas basicas del terreno',
            x=modelos_cl,
            y=[f1_base[m] for m in modelos_cl],
            marker_color=C['blue_light'],
            text=["{:.3f}".format(f1_base[m]) for m in modelos_cl],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Basico: %{y:.4f}<br>Comparable literatura (n=3799)<extra></extra>',
        ))
    if modelos_dl:
        fig.add_trace(go.Bar(
            name='Deep Learning',
            x=modelos_dl,
            y=[f1_dl[m] for m in modelos_dl],
            marker_color='#E5E7EB',
            text=["{:.3f}".format(f1_dl[m]) for m in modelos_dl],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>F1: %{y:.4f}<br>Deep Learning<extra></extra>',
        ))
    fig.add_hline(y=umbral_f1, line_dash='dash', line_color=C['dark'], line_width=1.5,
                  annotation_text='F1={:.2f}'.format(umbral_f1),
                  annotation_position='top right')
    fig.update_layout(**layout_base(
        title='El protocolo de evaluacion cambia el resultado',
        xlab='Modelo', ylab='F1-Score', height=460,
    ))
    fig.update_layout(barmode='group', yaxis_range=[0.3, 0.97])
    return fig


def fig_complejidad(modelos, umbral_f1):
    comp  = {'LR': 'Baja', 'SVM': 'Media', 'RF': 'Media',
             'ResNet-50': 'Alta', 'EfficientNet': 'Alta', 'U-Net': 'Alta'}
    orden = sorted(modelos, key=lambda m: F1_VALS.get(m, 0))
    tipos = [TIPO.get(m, '—') for m in orden]
    comps = [comp.get(m, '—') for m in orden]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=orden, x=[F1_VALS[m] for m in orden],
        orientation='h',
        marker_color=[col_modelo(m) for m in orden],
        text=["{:.3f}".format(F1_VALS[m]) for m in orden],
        textposition='outside',
        customdata=list(zip(tipos, comps)),
        hovertemplate='<b>%{y}</b><br>F1-Score: %{x:.4f}<br>Tipo: %{customdata[0]}<br>Complejidad: %{customdata[1]}<extra></extra>',
    ))
    fig.add_vline(x=umbral_f1, line_dash='dash', line_color=C['dark'], line_width=1.5,
                  annotation_text='F1={:.2f}'.format(umbral_f1),
                  annotation_position='top right',
                  annotation_font_color=C['dark'])
    fig.update_layout(**layout_base(
        title='Complejidad del modelo vs. resultado real',
        xlab='F1-Score — capacidad de deteccion',
        ylab='Modelo', height=max(350, len(orden) * 62),
    ))
    fig.update_xaxes(range=[0, 0.97])
    return fig


def fig_canales_colombia():
    canales = [
        ('S2-B7 RedEdge3', 0.8073, 'RedEdge', True,  'Sentinel-2 B7'),
        ('S2-B6 RedEdge2', 0.5625, 'RedEdge', True,  'Sentinel-2 B6'),
        ('ALOS DEM',       0.1954, 'Topo',    False, 'Modelo elevacion ALOS'),
        ('S1-VH SAR',      0.1882, 'SAR',     True,  'Sentinel-1 VH'),
        ('DEM Slope',      0.0430, 'Topo',    False, 'Pendiente DEM'),
        ('S2-B8A NIR-A',   0.0221, 'Optico',  True,  'Sentinel-2 B8A'),
    ]
    fig = go.Figure()
    grp_shown = set()
    for nm, delta, grp, disp, desc in canales:
        show   = grp not in grp_shown
        grp_shown.add(grp)
        estado = 'Copernicus (gratuito)' if disp else 'Requiere DEM externo'
        ht = '<b>{}</b><br>Delta: {:.4f}<br>Sensor: {}<br>Disponibilidad: {}<extra></extra>'.format(nm, delta, desc, estado)
        fig.add_trace(go.Bar(
            y=[nm], x=[delta], orientation='h',
            name=grp, legendgroup=grp, showlegend=show,
            marker_color=C[grp],
            hovertemplate=ht,
        ))
    fig.update_layout(**layout_base(
        title='Canales mas discriminativos y su disponibilidad para Colombia',
        xlab='Brecha de senal (Delta)',
        ylab='Canal satelital', height=380,
    ))
    fig.update_xaxes(range=[0, 0.97])
    return fig


def fig_consistencia(folds_dict, umbral_f1):
    orden  = sorted(folds_dict.keys(), key=lambda k: np.mean(folds_dict[k]), reverse=True)
    fig = go.Figure()
    for m in orden:
        vals  = folds_dict[m]
        media = np.mean(vals)
        std   = np.std(vals, ddof=1)
        color = C['red'] if 'RF' in m else '#9CA3AF'
        for i, v in enumerate(vals):
            ht = '<b>{}</b><br>Fold {}: {:.4f}<br>Media: {:.4f} | Std: {:.4f}<extra></extra>'.format(m, i+1, v, media, std)
            fig.add_trace(go.Scatter(
                x=[v], y=[m], mode='markers',
                name=m, legendgroup=m, showlegend=(i == 0),
                marker=dict(size=11, color=color, line=dict(color='white', width=1.5)),
                hovertemplate=ht,
            ))
        fig.add_trace(go.Scatter(
            x=[media - std, media + std], y=[m, m], mode='lines',
            line=dict(color=color, width=8), opacity=0.3,
            showlegend=False, hoverinfo='skip',
        ))
    fig.add_vline(x=umbral_f1, line_dash='dash', line_color=C['dark'], line_width=1.5,
                  annotation_text='F1={:.2f}'.format(umbral_f1),
                  annotation_position='bottom right',
                  annotation_font_color=C['dark'])
    fig.update_layout(**layout_base(
        title='Consistencia del modelo como indicador de confianza',
        xlab='F1-Score por fold — cada punto es un experimento independiente',
        ylab='Modelo', height=max(380, len(orden) * 72),
    ))
    return fig


# ── INICIO ────────────────────────────────────────────────────────
if seccion == "Inicio":
    st.markdown("# Deteccion de Deslizamientos con Machine Learning")
    st.markdown("### Que nos dicen los modelos sobre como proteger Colombia?")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    kpi_style = "font-size:2.2rem;font-weight:700;color:#DC2626"
    with c1:
        st.markdown('<div class="kpi-card"><div style="{}">400-600</div><div>eventos/ano en Colombia<br><small style="color:#9CA3AF">SGC, 2023</small></div></div>'.format(kpi_style), unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="kpi-card"><div style="{}">0</div><div>datasets etiquetados colombianos publicos</div></div>'.format(kpi_style), unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="kpi-card"><div style="{}">14</div><div>bandas satelitales analizadas<br>Sentinel-1/2 + DEM</div></div>'.format(kpi_style), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="arg-box"><strong>Argumento central:</strong><br>"El modelo mas sofisticado no siempre gana — y en Colombia, donde no hay un dataset propio, elegir mal el modelo y el protocolo puede ser la diferencia entre una herramienta util y una que falla cuando mas se necesita."</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Recorrido del analisis")
        pasos = [
            ("1. Exploracion",        "Que muestran los datos sin hipotesis?"),
            ("2. Trampa metodologica","El resultado cambia segun como evalues"),
            ("3. Colombia vs mundo",  "Donde estamos frente a la literatura?"),
            ("4. Complejidad",        "Mas parametros no garantizan mejor deteccion"),
            ("5. Disponibilidad",     "Que informacion satelital necesita Colombia?"),
            ("6. Confianza",          "En que modelo confiar con datos escasos?"),
        ]
        for paso, desc in pasos:
            st.markdown('<div class="hall-box"><strong>{}</strong> — {}</div>'.format(paso, desc), unsafe_allow_html=True)
    with col_b:
        st.markdown("#### Modelos evaluados")
        resumen = pd.DataFrame({
            'Modelo': TODOS,
            'Tipo':   [TIPO[m] for m in TODOS],
            'F1-Score': [F1_VALS[m] for m in TODOS],
        }).sort_values('F1-Score', ascending=False).reset_index(drop=True)
        st.dataframe(resumen, use_container_width=True, hide_index=True)

# ── EXPLORACION ──────────────────────────────────────────────────
elif seccion == "Exploracion":
    st.markdown("# Fase Exploratoria — Donde estamos?")
    st.caption("Pasa el cursor para ver valores exactos · Clic en leyenda para mostrar/ocultar · Arrastra para zoom")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "H1 - F1 por modelo", "H2 - Precision vs Cobertura",
        "H3 - Canales satelitales", "H4 - Brecha de senal", "H5 - Variabilidad folds",
    ])
    modelos_vis = [m for m in TODOS if m in sel]

    with tab1:
        st.markdown("**Que tan bien detecta cada modelo un deslizamiento?**")
        if modelos_vis:
            st.plotly_chart(fig_f1_barras(modelos_vis, umbral), use_container_width=True)
        else:
            st.warning("Selecciona al menos un modelo.")

    with tab2:
        st.markdown("**Precision o cobertura? Podemos tener los dos?**")
        st.plotly_chart(fig_pr_scatter(modelos_vis, mostrar_lit, umbral), use_container_width=True)

    with tab3:
        st.markdown("**Que informacion satelital separa mejor las clases?**")
        st.plotly_chart(fig_canales(), use_container_width=True)

    with tab4:
        st.markdown("**Como se diferencian las clases en los canales clave?**")
        st.plotly_chart(fig_dot_plot(), use_container_width=True)

    with tab5:
        st.markdown("**El rendimiento es estable entre experimentos?**")
        folds_vis = {k: v for k, v in fold_data.items()
                     if any(k.startswith(m.split()[0]) for m in sel)} or fold_data
        st.plotly_chart(fig_folds(folds_vis, umbral), use_container_width=True)

# ── ARGUMENTO ────────────────────────────────────────────────────
elif seccion == "Argumento":
    st.markdown("# Fase Aclaratoria — Del hallazgo al argumento")
    st.caption("Pasa el cursor sobre los elementos para ver valores · Clic en leyenda para aislar series")
    st.markdown('<div class="arg-box">Hilo conductor: <strong>Problema Colombia &rarr; que modelo responde?</strong></div>', unsafe_allow_html=True)
    st.markdown("---")

    taba1, taba2, taba3, taba4, taba5 = st.tabs([
        "A1 - Trampa metodologica", "A2 - Colombia vs mundo",
        "A3 - Complejidad != rendimiento", "A4 - Que necesita Colombia", "A5 - Confianza",
    ])
    modelos_vis = [m for m in TODOS if m in sel]
    m_cl = [m for m in ['LR', 'SVM', 'RF'] if m in sel]
    m_dl = [m for m in ['ResNet-50', 'EfficientNet', 'U-Net'] if m in sel]

    with taba1:
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.markdown("### El resultado cambia segun como evalues el modelo")
            st.markdown('<div class="arg-box">Si se reporta solo el protocolo mas favorable, se puede <strong>sobrestimar el rendimiento</strong> en condiciones locales.</div>', unsafe_allow_html=True)
            st.markdown("**14 bandas:** protocolo optimizado, n=1500")
            st.markdown("**Basicas:** comparable a literatura, n=3799")
        with cc2:
            st.plotly_chart(fig_protocolos(m_cl, m_dl, umbral), use_container_width=True)

    with taba2:
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.markdown("### Donde estamos frente a la literatura internacional?")
            st.markdown('<div class="arg-box">Los modelos de referencia se entrenaron en Nepal, Peru e Italia. Colombia tiene <strong>terreno andino diferente</strong> y ningun dataset etiquetado.</div>', unsafe_allow_html=True)
        with cc2:
            st.plotly_chart(fig_pr_scatter(modelos_vis, mostrar_lit, umbral), use_container_width=True)

    with taba3:
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.markdown("### Complejidad del modelo vs. resultado real")
            st.markdown('<div class="arg-box">U-Net — la arquitectura mas compleja — tiene el <strong>peor F1</strong>. Mas parametros no garantizan mejor deteccion.</div>', unsafe_allow_html=True)
        with cc2:
            st.plotly_chart(fig_complejidad(modelos_vis, umbral), use_container_width=True)

    with taba4:
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.markdown("### Que informacion satelital necesita Colombia?")
            st.markdown('<div class="arg-box">Las bandas <strong>RedEdge</strong> (Sentinel-2) son las mas discriminativas y estan disponibles gratuitamente. El desafio: <strong>etiquetas post-evento</strong>.</div>', unsafe_allow_html=True)
        with cc2:
            st.plotly_chart(fig_canales_colombia(), use_container_width=True)

    with taba5:
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.markdown("### En cual modelo confiar con datos escasos?")
            st.markdown('<div class="arg-box">La <strong>consistencia entre folds</strong> es el indicador mas importante cuando los datos son pocos y de un solo evento.</div>', unsafe_allow_html=True)
            st.markdown("**RF:** Std=0.008 — el mas consistente")
            st.markdown("**SVM:** Std=0.033 — mayor variabilidad")
        with cc2:
            folds_a5 = {k: v for k, v in fold_data.items()
                        if any(k.startswith(m.split()[0]) for m in sel)} or fold_data
            st.plotly_chart(fig_consistencia(folds_a5, umbral), use_container_width=True)

# ── CONCLUSION ───────────────────────────────────────────────────
elif seccion == "Conclusion":
    st.markdown("# Conclusion — Lo que necesita Colombia")
    st.markdown('<div class="arg-box" style="font-size:1.05rem">La arquitectura viene despues del contexto. Random Forest — interpretable y eficiente con pocos datos — es un punto de partida mas solido que redes profundas disenadas para miles de imagenes segmentadas.</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Sintesis de hallazgos")
        tabla = pd.DataFrame({
            'Hallazgo': ['F1 por modelo', 'Precision vs Cobertura', 'Canales satelitales',
                         'Brecha de senal', 'Variabilidad folds'],
            'Observacion': [
                'RF supera F1=0.80; U-Net queda muy por debajo',
                'RF prioriza cobertura (Recall=0.96)',
                'Bandas RedEdge dominan la discriminacion',
                'RedEdge3 tiene brecha 4x mayor que SAR-VH',
                'RF es el mas consistente (Std=0.008)',
            ],
        })
        st.dataframe(tabla, use_container_width=True, hide_index=True)

        st.markdown("### Tabla comparativa")
        df_s = df.copy()
        for col in ['F1 medio', 'Precision', 'Recall']:
            if col in df_s.columns:
                df_s[col] = df_s[col].map(lambda x: '{:.4f}'.format(x) if pd.notna(x) else '—')
        cols_show = [c for c in ['Modelo', 'Tipo', 'F1 medio', 'Precision', 'Recall'] if c in df_s.columns]
        st.dataframe(df_s[cols_show], use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Condiciones para Colombia")
        condiciones = [
            ("Dataset etiquetado nacional", "No existe",
             "#FEE2E2", "#DC2626", "Sin esto, cualquier modelo es extrapolacion"),
            ("Bandas satelitales", "Sentinel-2 disponible (RedEdge incluido)",
             "#DCFCE7", "#16A34A", "La senal esta — faltan etiquetas post-evento"),
            ("Protocolo de evaluacion", "Depende del estudio",
             "#FEF9C3", "#CA8A04", "Usar 5 folds, comparable a literatura"),
            ("Arquitectura apropiada", "RF como punto de partida",
             "#DCFCE7", "#16A34A", "Interpretable, eficiente con pocos datos"),
        ]
        for cond, estado, bg, bc, impl in condiciones:
            card = (
                "<div style='background:{bg};border-left:4px solid {bc};"
                "padding:10px 14px;border-radius:4px;margin-bottom:9px'>"
                "<strong style='color:{bc}'>{cond}</strong><br>"
                "<span style='color:#374151'>{estado}</span><br>"
                "<small style='color:#6B7280'>{impl}</small></div>"
            ).format(bg=bg, bc=bc, cond=cond, estado=estado, impl=impl)
            st.markdown(card, unsafe_allow_html=True)

        st.markdown("### Proximos pasos")
        st.markdown("""
1. **Etiquetar** imagenes historicas del SGC (Servicio Geologico Colombiano)
2. **Descargar** bandas RedEdge de Sentinel-2 via API Copernicus
3. **Entrenar y evaluar** con 5 folds y protocolo comparable a literatura
        """)

    st.markdown("---")
    st.caption("Datos: Landslide4Sense | Benchmarks: Ghorbanzadeh (2022), L4S (2022), Liu et al. (2024), Enhanced U-Net++ (2025)")
