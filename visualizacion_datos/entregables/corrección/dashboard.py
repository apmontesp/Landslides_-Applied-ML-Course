"""
Dashboard interactivo — Detección de Deslizamientos con ML en Colombia
Visualización de Datos · Universidad EAFIT · Ana Patricia Montes · 2025
"""
import os, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Deslizamientos ML — Colombia",
    page_icon="🏔",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main { background:#FFFFFF; }
  .block-container { padding-top:1.4rem; padding-bottom:2rem; }
  h1 { color:#1E3A5F; }
  h2 { color:#1E3A5F; border-bottom:2px solid #DC2626; padding-bottom:5px; }
  .arg-box  { background:#FEF2F2; border-left:4px solid #DC2626;
              padding:11px 15px; border-radius:4px; margin-bottom:.9rem;
              font-size:.94rem; color:#7F1D1D; }
  .hall-box { background:#F9FAFB; border-left:4px solid #6B7280;
              padding:9px 13px; border-radius:4px; margin-bottom:.7rem;
              font-size:.9rem; color:#374151; }
  .kpi-card { background:#F9FAFB; border:1px solid #E5E7EB;
              border-radius:8px; padding:14px; text-align:center; }
  .frase-final { background:#1E3A5F; color:#FFFFFF; border-radius:10px;
                 padding:22px 28px; font-size:1.15rem; font-style:italic;
                 text-align:center; margin-top:2rem; line-height:1.7; }
</style>
""", unsafe_allow_html=True)

# ── DATOS ──────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, '..', '..', 'data')

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'comparison_table.csv'))
    for col in ['F1 medio', 'Std', 'AUC-ROC', 'Precisión', 'Recall', 'IoU']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    FOLDS_FB = {
        'LR':  [0.7929, 0.7681, 0.8232, 0.7772, 0.7813],
        'SVM': [0.7526, 0.7731, 0.8100, 0.8325, 0.8189],
        'RF':  [0.8363, 0.8238, 0.8480, 0.8401, 0.8361],
        'ResNet-50':      [0.7762, 0.7708, 0.8154, 0.7636, 0.8065],
        'U-Net ResNet34': [0.6855, 0.7084, 0.7061, 0.6900, 0.6842],
    }
    try:
        def lf(fname, key='best_f1', alt=None):
            with open(os.path.join(DATA_DIR, 'folds', fname)) as fh:
                d = json.load(fh)
            return [x.get(key, x.get(alt, 0)) if alt else x[key] for x in d['folds']]
        folds = {
            'LR':             lf('logistic_regression_folds.json'),
            'SVM':            lf('svm_folds.json'),
            'RF':             lf('random_forest_folds.json'),
            'ResNet-50':      lf('resnet50_folds.json', key='f1_thr05'),
            'U-Net ResNet34': lf('unet_folds.json', key='f1_pixel_thr05'),
        }
    except Exception:
        folds = FOLDS_FB
    return df, folds

df, fold_data = load_data()

# ── PALETA ─────────────────────────────────────────────────────────
C = {
    'red':        '#DC2626',
    'navy':       '#1E3A5F',
    'gray':       '#9CA3AF',
    'dark':       '#374151',
    'clasico':    '#374151',
    'dl':         '#C4B5FD',
    'blue_dark':  '#1E3A5F',
    'blue_light': '#93C5FD',
    'RedEdge':    '#7C3AED',
    'Topo':       '#D97706',
    'SAR':        '#0369A1',
    'Optico':     '#059669',
    'bg':         'white',
    'grid':       '#EEEEEE',
}

TODOS   = ['LR', 'SVM', 'RF', 'ResNet-50', 'EfficientNet', 'U-Net']
F1_VALS = {'LR': 0.7886, 'SVM': 0.7974, 'RF': 0.8368,
           'ResNet-50': 0.7840, 'EfficientNet': 0.7554, 'U-Net': 0.4443}
TIPO    = {'LR': 'Clásico', 'SVM': 'Clásico', 'RF': 'Clásico',
           'ResNet-50': 'Deep Learning', 'EfficientNet': 'Deep Learning', 'U-Net': 'Deep Learning'}

def col_modelo(m):
    if m == 'RF':      return C['red']
    if m in ['SVM','LR']: return C['clasico']
    return C['dl']

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

# ── SIDEBAR ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Landslide ML · Colombia")
    st.markdown("---")
    seccion = st.radio("Navegación",
                       ["Inicio", "Exploración", "Análisis aclaratorio", "Conclusión"],
                       index=0)
    st.markdown("---")
    st.markdown("**Filtros**")
    umbral = st.slider("Umbral F1 de referencia", 0.60, 0.95, 0.80, 0.01)
    sel    = st.multiselect("Modelos a mostrar", TODOS, default=TODOS)
    mostrar_lit = st.toggle("Mostrar benchmarks literatura", value=True)
    st.markdown("---")
    st.caption("Dataset: Landslide4Sense (Nepal, Perú, Italia)")

# ══════════════════════════════════════════════════════════════════
# INICIO
# ══════════════════════════════════════════════════════════════════
if seccion == "Inicio":
    st.markdown("# Detección de Deslizamientos con Machine Learning")
    st.markdown("### ¿Qué modelo usar como punto de partida en Colombia?")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    kpi = "font-size:2.2rem;font-weight:700;color:#DC2626"
    with c1:
        st.markdown(f'<div class="kpi-card"><div style="{kpi}">400-600</div><div>eventos/año en Colombia<br><small style="color:#9CA3AF">SGC, 2023</small></div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-card"><div style="{kpi}">0</div><div>datasets etiquetados<br>colombianos</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-card"><div style="{kpi}">14</div><div>bandas satelitales<br>Sentinel-1/2 + DEM</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-card"><div style="{kpi}">5</div><div>modelos evaluados<br>ML + Deep Learning</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="arg-box"><strong>Pregunta central:</strong> Este trabajo no entrena un modelo colombiano — Colombia no tiene datos para eso. Evalúa cuál modelo entrenado con datos internacionales (Nepal, Perú, Italia) sería el punto de partida más adecuado si se aplicara en Colombia.</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Estructura del análisis")
        pasos = [
            ("Exploración H1", "F1-Score por modelo — ¿quién detecta mejor?"),
            ("Exploración H2", "Precisión vs. Cobertura — trade-off de errores"),
            ("Exploración H3", "Canales satelitales — ¿qué señal importa más?"),
            ("Exploración H4", "Variabilidad — ¿es estable el resultado?"),
            ("Aclaratorio A1", "¿Los resultados de la literatura son comparables?"),
            ("Aclaratorio A2", "¿Qué tan grande es la brecha frente al estado del arte?"),
            ("Aclaratorio A3", "¿Vale la pena invertir en arquitectura compleja?"),
            ("Aclaratorio A4", "¿Qué modelo elegir como punto de partida?"),
        ]
        for paso, desc in pasos:
            st.markdown(f'<div class="hall-box"><strong>{paso}</strong> — {desc}</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown("#### Modelos evaluados")
        resumen = pd.DataFrame({
            'Modelo': TODOS,
            'Tipo':   [TIPO[m] for m in TODOS],
            'F1-Score': [F1_VALS[m] for m in TODOS],
        }).sort_values('F1-Score', ascending=False).reset_index(drop=True)
        st.dataframe(resumen, use_container_width=True, hide_index=True)
        st.caption("Datos: Landslide4Sense — no hay datos colombianos propios")

# ══════════════════════════════════════════════════════════════════
# EXPLORACIÓN
# ══════════════════════════════════════════════════════════════════
elif seccion == "Exploración":
    st.markdown("# Exploración — ¿Qué muestran los datos?")
    st.caption("Pasa el cursor para valores exactos · Clic en leyenda para ocultar/mostrar · Arrastra para zoom")
    st.markdown('<div class="hall-box">Esta sección describe los datos y resultados sin hipótesis. No hay interpretaciones sobre Colombia — eso es tarea del análisis aclaratorio.</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "H1 — F1 por modelo", "H2 — Precisión vs Cobertura",
        "H3 — Canales satelitales", "H4 — Variabilidad entre folds",
    ])
    modelos_vis = [m for m in TODOS if m in sel]

    with tab1:
        st.markdown("**¿Qué tan bien detecta cada modelo un deslizamiento?**")
        if modelos_vis:
            orden  = sorted(modelos_vis, key=lambda m: F1_VALS.get(m, 0))
            colors = [col_modelo(m) for m in orden]
            f1s    = [F1_VALS[m] for m in orden]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=orden, x=f1s, orientation='h', marker_color=colors,
                text=[f"{v:.3f}" for v in f1s], textposition='outside',
                customdata=[TIPO[m] for m in orden],
                hovertemplate='<b>%{y}</b><br>F1: %{x:.4f}<br>Tipo: %{customdata}<extra></extra>',
            ))
            fig.add_vline(x=umbral, line_dash='dash', line_color=C['dark'], line_width=1.5,
                          annotation_text=f'F1={umbral:.2f}', annotation_position='top right')
            fig.update_layout(**layout_base('H1 — F1-Score por modelo (5-fold, ordenado mayor→menor)',
                                            'F1-Score', 'Modelo', max(350, len(orden)*60)))
            fig.update_xaxes(range=[0, 0.97])
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("**¿Precisión o cobertura? En detección de desastres, perder un evento real cuesta más que una falsa alarma.**")
        datos_pr = {'LR':(0.7971,0.7806,'Clásico'),'SVM':(0.8193,0.7777,'Clásico'),
                    'RF':(0.7439,0.9569,'Clásico'),'ResNet-50':(0.7219,0.8771,'Deep Learning')}
        lit = [('Ghorbanzadeh et al. (2022)',0.717,'#FECACA'),
               ('Lv et al. — L4S (2022)',0.739,'#F87171'),
               ('Liu et al. — Multi-scale (2024)',0.760,'#EF4444'),
               ('Enhanced U-Net++ (2025)',0.841,'#B91C1C')]
        fig = go.Figure()
        if mostrar_lit:
            r_arr = np.linspace(0.63, 0.999, 300)
            for nombre, f1v, cl in lit:
                p_arr = f1v * r_arr / (2*r_arr - f1v)
                mask  = (p_arr > 0) & (p_arr <= 1.0)
                fig.add_trace(go.Scatter(x=r_arr[mask], y=p_arr[mask], mode='lines',
                    line=dict(color=cl, dash='dash', width=1.3), name=nombre,
                    hovertemplate=f'<b>{nombre}</b><br>F1={f1v}<br>Recall: %{{x:.3f}}<br>Precisión: %{{y:.3f}}<extra></extra>'))
        for nm, (prec, rec, tipo) in datos_pr.items():
            if nm not in modelos_vis: continue
            mk = 'diamond' if tipo == 'Deep Learning' else 'circle'
            fig.add_trace(go.Scatter(x=[rec], y=[prec], mode='markers+text', name=nm,
                marker=dict(size=14, color=col_modelo(nm), line=dict(color='white',width=2), symbol=mk),
                text=[nm], textposition='top right', textfont=dict(size=11, color=col_modelo(nm)),
                hovertemplate=f'<b>{nm}</b><br>Recall: {rec:.4f}<br>Precisión: {prec:.4f}<extra></extra>'))
        fig.update_layout(**layout_base('H2 — Precisión vs. Cobertura',
            'Cobertura (Recall)', 'Precisión', 500))
        fig.update_xaxes(range=[0.62, 1.02]); fig.update_yaxes(range=[0.62, 0.90])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("**¿Cuáles de las 14 bandas separan mejor los píxeles con y sin deslizamiento?**")
        dot_data = [
            ('S2-B7 RedEdge3',2.0209,1.2136,'RedEdge'),('S2-B6 RedEdge2',1.4782,0.9157,'RedEdge'),
            ('ALOS DEM',1.2739,1.0786,'Topo'),('S1-VH SAR',1.2488,1.0606,'SAR'),
            ('S2-B8A NIR-A',1.0397,1.0176,'Optico'),('DEM Slope',1.0703,1.0274,'Topo'),
        ]
        fig = go.Figure()
        grp_shown = set()
        for nm, pos, neg, grp in dot_data:
            show = grp not in grp_shown; grp_shown.add(grp)
            ht = f'<b>{nm}</b><br>Con desliz.: {pos:.4f}<br>Sin desliz.: {neg:.4f}<br>Δ={pos-neg:.4f}<extra></extra>'
            fig.add_trace(go.Scatter(x=[neg,pos], y=[nm,nm], mode='lines',
                line=dict(color='#D1D5DB',width=2), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=[pos], y=[nm], mode='markers',
                name=grp, legendgroup=grp, showlegend=show,
                marker=dict(size=12, color=C[grp], line=dict(color='white',width=2)),
                hovertemplate=ht))
            fig.add_trace(go.Scatter(x=[neg], y=[nm], mode='markers', showlegend=False,
                marker=dict(size=12, color='white', line=dict(color=C[grp],width=2.5)),
                hovertemplate=ht))
        fig.update_layout(**layout_base('H3 — Poder discriminativo de canales satelitales',
            'Reflectancia media normalizada', 'Canal satelital', 420))
        fig.update_xaxes(range=[0.8, 2.55])
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("**¿El rendimiento de cada modelo es estable o depende del subconjunto de datos?**")
        folds_vis = {k:v for k,v in fold_data.items()
                     if any(k.startswith(m.split()[0]) for m in sel)} or fold_data
        orden = sorted(folds_vis.keys(), key=lambda k: np.median(folds_vis[k]))
        fig = go.Figure()
        for m in orden:
            vals  = folds_vis[m]
            media = np.mean(vals); std = np.std(vals, ddof=1)
            color = C['red'] if 'RF' in m else '#9CA3AF'
            ht = f'<b>{m}</b><br>F1: %{{x:.4f}}<br>Media: {media:.4f} | Std: {std:.4f}<extra></extra>'
            fig.add_trace(go.Box(x=vals, y=[m]*len(vals), orientation='h', name=m,
                boxpoints='all', jitter=0.4, pointpos=0,
                marker=dict(size=9, color=color, line=dict(color='white',width=1.5)),
                line=dict(color=color),
                fillcolor='rgba({},{},{},0.25)'.format(int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)),
                hovertemplate=ht))
        fig.add_vline(x=umbral, line_dash='dash', line_color=C['dark'], line_width=1.5,
                      annotation_text=f'F1={umbral:.2f}', annotation_position='bottom right')
        fig.update_layout(**layout_base('H4 — Variabilidad entre experimentos (5 folds)',
            'F1-Score por fold', 'Modelo', max(380, len(orden)*70)))
        fig.update_traces(boxmean=True)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# ANÁLISIS ACLARATORIO
# ══════════════════════════════════════════════════════════════════
elif seccion == "Análisis aclaratorio":
    st.markdown("# Análisis Aclaratorio — Del hallazgo a la decisión")
    st.caption("Cada pestaña responde una pregunta de negocio con implicación concreta para Colombia")
    st.markdown('<div class="arg-box">Los modelos se entrenaron con datos de Nepal, Perú e Italia. El análisis evalúa cuál sería el mejor punto de partida para Colombia, donde no existen datos etiquetados propios.</div>', unsafe_allow_html=True)
    st.markdown("---")

    taba1, taba2, taba3, taba4 = st.tabs([
        "A1 — Protocolo de evaluación", "A2 — Brecha Colombia vs. literatura",
        "A3 — Complejidad vs. resultado", "A4 — ¿Qué modelo elegir?",
    ])
    modelos_vis = [m for m in TODOS if m in sel]
    m_cl = [m for m in ['LR','SVM','RF'] if m in sel]
    m_dl = [m for m in ['ResNet-50','EfficientNet','U-Net'] if m in sel]

    with taba1:
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.markdown("### ¿Los resultados de la literatura son comparables con los nuestros?")
            st.markdown('<div class="arg-box">El mismo modelo puede reportar un F1 muy distinto según el protocolo usado. Comparar sin verificar el protocolo equivale a comparar cosas distintas.</div>', unsafe_allow_html=True)
            st.markdown("**14 bandas:** protocolo optimizado, n=1500")
            st.markdown("**HOG+DEM+NDVI:** comparable con literatura, n=3799")
        with cc2:
            f1_opt  = {'LR':0.7886,'SVM':0.7974,'RF':0.8368}
            f1_base = {'LR':0.7512,'SVM':0.7340,'RF':0.7891}
            f1_dl   = {'ResNet-50':0.7840,'EfficientNet':0.7554,'U-Net':0.4443}
            fig = go.Figure()
            if m_cl:
                fig.add_trace(go.Bar(name='14 bandas del satélite', x=m_cl,
                    y=[f1_opt[m] for m in m_cl], marker_color=C['blue_dark'],
                    text=[f"{f1_opt[m]:.3f}" for m in m_cl], textposition='outside'))
                fig.add_trace(go.Bar(name='Características básicas (HOG+DEM+NDVI)', x=m_cl,
                    y=[f1_base[m] for m in m_cl], marker_color=C['blue_light'],
                    text=[f"{f1_base[m]:.3f}" for m in m_cl], textposition='outside'))
            if m_dl:
                fig.add_trace(go.Bar(name='Deep Learning', x=m_dl,
                    y=[f1_dl[m] for m in m_dl], marker_color='#E5E7EB',
                    text=[f"{f1_dl[m]:.3f}" for m in m_dl], textposition='outside'))
            fig.add_hline(y=umbral, line_dash='dash', line_color=C['dark'], line_width=1.5,
                          annotation_text=f'F1={umbral:.2f}', annotation_position='top right')
            fig.update_layout(**layout_base('A1 — El protocolo cambia el resultado',
                'Modelo', 'F1-Score', 460))
            fig.update_layout(barmode='group', yaxis_range=[0.3, 0.97])
            st.plotly_chart(fig, use_container_width=True)

    with taba2:
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.markdown("### ¿Qué tan grande es la brecha frente al estado del arte?")
            st.markdown('<div class="arg-box">Los benchmarks internacionales se entrenaron en Nepal, Perú e Italia. Aplicarlos en Colombia sin reentrenamiento introduce una brecha — pero RF la reduce al alcanzar el benchmark 2025 en cobertura.</div>', unsafe_allow_html=True)
        with cc2:
            datos_pr = {'LR':(0.7971,0.7806),'SVM':(0.8193,0.7777),
                        'RF':(0.7439,0.9569),'ResNet-50':(0.7219,0.8771)}
            lit = [('Ghorbanzadeh et al. (2022)',0.717,'#FECACA'),
                   ('Lv et al. — L4S (2022)',0.739,'#F87171'),
                   ('Liu et al. (2024)',0.760,'#EF4444'),
                   ('Enhanced U-Net++ (2025)',0.841,'#B91C1C')]
            fig = go.Figure()
            if mostrar_lit:
                r_arr = np.linspace(0.63, 0.999, 300)
                for nombre, f1v, cl in lit:
                    p_arr = f1v * r_arr / (2*r_arr - f1v)
                    mask  = (p_arr > 0) & (p_arr <= 1.0)
                    fig.add_trace(go.Scatter(x=r_arr[mask], y=p_arr[mask], mode='lines',
                        line=dict(color=cl, dash='dash', width=1.3), name=nombre))
            for nm, (prec, rec) in datos_pr.items():
                if nm not in modelos_vis: continue
                mk = 'diamond' if TIPO[nm]=='Deep Learning' else 'circle'
                fig.add_trace(go.Scatter(x=[rec], y=[prec], mode='markers+text', name=nm,
                    marker=dict(size=14, color=col_modelo(nm), line=dict(color='white',width=2), symbol=mk),
                    text=[nm], textposition='top right',
                    textfont=dict(size=11, color=col_modelo(nm))))
            fig.update_layout(**layout_base('A2 — Landslide4Sense vs. benchmarks internacionales',
                'Cobertura (Recall)', 'Precisión', 500))
            fig.update_xaxes(range=[0.62, 1.02]); fig.update_yaxes(range=[0.62, 0.90])
            st.plotly_chart(fig, use_container_width=True)

    with taba3:
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.markdown("### ¿Vale la pena invertir en arquitectura compleja para Colombia?")
            st.markdown('<div class="arg-box">U-Net — la arquitectura más compleja — tiene el peor F1. Complejidad alta no garantiza mejor detección. Para Colombia, con recursos limitados, esto importa.</div>', unsafe_allow_html=True)
            comp = {'LR':'Baja','SVM':'Media','RF':'Media',
                    'ResNet-50':'Alta','EfficientNet':'Alta','U-Net':'Alta'}
        with cc2:
            orden = sorted(modelos_vis, key=lambda m: F1_VALS.get(m,0))
            fig = go.Figure()
            fig.add_trace(go.Bar(y=orden, x=[F1_VALS[m] for m in orden], orientation='h',
                marker_color=[col_modelo(m) for m in orden],
                text=[f"{F1_VALS[m]:.3f}" for m in orden], textposition='outside',
                customdata=[(TIPO[m], comp.get(m,'—')) for m in orden],
                hovertemplate='<b>%{y}</b><br>F1: %{x:.4f}<br>Tipo: %{customdata[0]}<br>Complejidad: %{customdata[1]}<extra></extra>'))
            fig.add_vline(x=umbral, line_dash='dash', line_color=C['dark'], line_width=1.5,
                          annotation_text=f'F1={umbral:.2f}', annotation_position='top right')
            fig.update_layout(**layout_base('A3 — Complejidad del modelo vs. resultado real',
                'F1-Score', 'Modelo', max(350, len(orden)*62)))
            fig.update_xaxes(range=[0, 0.97])
            st.plotly_chart(fig, use_container_width=True)

    with taba4:
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.markdown("### ¿Qué modelo elegir como punto de partida en Colombia?")
            st.markdown('<div class="arg-box">Sin datos colombianos propios, la consistencia y la interpretabilidad pesan tanto como el F1. El modelo más estable en condiciones no vistas es el más confiable.</div>', unsafe_allow_html=True)
            st.markdown("**Criterios evaluados:**")
            for c_txt in ["F1-Score — rendimiento empírico",
                          "Consistencia — estabilidad entre folds",
                          "Costo bajo — viabilidad con infraestructura local",
                          "Interpretabilidad — transparencia para tomadores de decisión"]:
                st.markdown(f"- {c_txt}")
        with cc2:
            modelos_sc = ['RF','SVM','LR','ResNet-50','EfficientNet','U-Net']
            criterios  = ['F1-Score','Consistencia','Costo bajo','Interpretabilidad']
            scores_sc  = {
                'RF':          [0.837, 0.95, 0.90, 0.90],
                'SVM':         [0.797, 0.70, 0.85, 0.70],
                'LR':          [0.789, 0.81, 0.95, 0.80],
                'ResNet-50':   [0.784, 0.85, 0.30, 0.25],
                'EfficientNet':[0.755, 0.82, 0.25, 0.20],
                'U-Net':       [0.444, 0.78, 0.20, 0.15],
            }
            col_crit   = [C['blue_dark'], '#2563EB', C['blue_light'], '#BFDBFE']
            fig = go.Figure()
            for j, (crit, cc) in enumerate(zip(criterios, col_crit)):
                vals = [scores_sc[m][j] for m in modelos_sc if m in modelos_vis or m == 'RF']
                mods = [m for m in modelos_sc if m in modelos_vis or m == 'RF']
                fig.add_trace(go.Bar(name=crit, x=mods, y=vals, marker_color=cc,
                    text=[f"{v:.2f}" for v in vals], textposition='outside',
                    hovertemplate=f'<b>%{{x}}</b><br>{crit}: %{{y:.2f}}<extra></extra>'))
            fig.add_vline(x=2.5, line_dash='dash', line_color='#6B7280', line_width=1.5)
            fig.add_annotation(x=1.0, y=1.1, text='Clásicos', showarrow=False,
                               font=dict(size=11, color='#374151'))
            fig.add_annotation(x=4.0, y=1.1, text='Deep Learning', showarrow=False,
                               font=dict(size=11, color='#374151'))
            fig.update_layout(**layout_base('A4 — Perfil de decisión por modelo',
                'Modelo', 'Puntuación (0=peor · 1=mejor)', 480))
            fig.update_layout(barmode='group', yaxis_range=[0, 1.18])
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# CONCLUSIÓN
# ══════════════════════════════════════════════════════════════════
elif seccion == "Conclusión":
    st.markdown("# Conclusión — Lo que necesita Colombia")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Síntesis de hallazgos")
        tabla_h = pd.DataFrame({
            'Hallazgo': ['H1 — F1 por modelo','H2 — Precisión/Recall',
                         'H3 — Canales','H4 — Variabilidad'],
            'Observación': [
                'RF lidera (0.837); U-Net es el peor (0.444)',
                'RF prioriza cobertura (Recall=0.96)',
                'RedEdge3 tiene Δ=0.807 — 4× mayor que SAR-VH',
                'RF es el más consistente (Std=0.008)',
            ],
        })
        st.dataframe(tabla_h, use_container_width=True, hide_index=True)

        st.markdown("### Síntesis del argumento")
        tabla_a = pd.DataFrame({
            'Pregunta': ['A1 — Protocolo','A2 — Brecha','A3 — Complejidad','A4 — Decisión'],
            'Implicación': [
                'Usar 5-fold geoespacial para comparar con literatura',
                'RF alcanza benchmark 2025 en cobertura',
                'Clásicos igualan a DL con mucho menor costo',
                'RF: mejor perfil global para Colombia',
            ],
        })
        st.dataframe(tabla_a, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Condiciones para implementar en Colombia")
        condiciones = [
            ("✗  Dataset etiquetado nacional", "No existe aún",
             "#FEE2E2","#DC2626","Sin esto, cualquier modelo es extrapolación"),
            ("✓  Señal satelital disponible", "Sentinel-2 RedEdge — Copernicus gratuito",
             "#DCFCE7","#16A34A","La señal está — faltan las etiquetas"),
            ("⚠  Protocolo de evaluación", "Usar 5-fold con separación geoespacial",
             "#FEF9C3","#CA8A04","Comparable con literatura internacional"),
            ("✓  Modelo recomendado", "Random Forest — estable, interpretable, adaptable",
             "#DCFCE7","#16A34A","Punto de partida más sólido para Colombia"),
        ]
        for cond, estado, bg, bc, impl in condiciones:
            st.markdown(
                f"<div style='background:{bg};border-left:4px solid {bc};"
                f"padding:10px 14px;border-radius:4px;margin-bottom:9px'>"
                f"<strong style='color:{bc}'>{cond}</strong><br>"
                f"<span style='color:#374151'>{estado}</span><br>"
                f"<small style='color:#6B7280'>{impl}</small></div>",
                unsafe_allow_html=True)

        st.markdown("### Próximos pasos")
        for paso in [
            "Etiquetar imágenes históricas del SGC con ubicación geográfica precisa",
            "Descargar bandas RedEdge de Sentinel-2 vía API Copernicus",
            "Entrenar y evaluar con 5-fold geoespacial, protocolo comparable a literatura",
        ]:
            st.markdown(f"- {paso}")

    st.markdown("---")
    st.markdown(
        '<div class="frase-final">'
        '"No esperamos tener datos perfectos para empezar.<br>'
        'Empezamos para tener datos."'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("  ")
    st.caption("Dataset: Landslide4Sense | Benchmarks: Ghorbanzadeh (2022), L4S Competition (2022), Liu et al. (2024), Enhanced U-Net++ (2025)")
