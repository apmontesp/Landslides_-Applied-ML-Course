"""
Dashboard — Detección de Deslizamientos con Machine Learning en Colombia
Visualización de Datos · Entregable Final
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import streamlit as st

# ──────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deslizamientos ML · Colombia",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado
st.markdown("""
<style>
    .main { background-color: #FFFFFF; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { color: #111827; font-size: 1.7rem !important; }
    h2 { color: #1F2937; font-size: 1.25rem !important; border-bottom: 2px solid #DC2626;
         padding-bottom: 6px; margin-top: 1.2rem; }
    h3 { color: #374151; font-size: 1.05rem !important; }
    .argumento-box {
        background: #FEF2F2; border-left: 4px solid #DC2626;
        padding: 12px 16px; border-radius: 4px; margin-bottom: 1rem;
        font-size: 0.95rem; color: #7F1D1D;
    }
    .hallazgo-box {
        background: #F9FAFB; border-left: 4px solid #6B7280;
        padding: 10px 14px; border-radius: 4px; margin-bottom: 0.8rem;
        font-size: 0.9rem; color: #374151;
    }
    .metric-card {
        background: #F9FAFB; border: 1px solid #E5E7EB;
        border-radius: 8px; padding: 14px; text-align: center;
    }
    .stSidebar { background-color: #F9FAFB; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# RUTAS Y DATOS
# ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, '..', 'data')

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'comparison_table.csv'))
    for col in ['F1 medio','Std','AUC-ROC','Precisión','Recall','IoU']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    ch = pd.read_csv(os.path.join(DATA_DIR, 'channel_stats_by_class.csv'))

    FOLDS_FB = {
        'LR':  [0.7929, 0.7681, 0.8232, 0.7772, 0.7813],
        'SVM': [0.7526, 0.7731, 0.8100, 0.8325, 0.8189],
        'RF':  [0.8363, 0.8238, 0.8480, 0.8401, 0.8361],
        'ResNet-50':      [0.7762, 0.7708, 0.8154, 0.7636, 0.8065],
        'U-Net ResNet34': [0.6855, 0.7084, 0.7061, 0.6900, 0.6842],
    }
    try:
        def lf(fname, key='best_f1', alt='f1_pixel_thr05'):
            with open(os.path.join(DATA_DIR, 'folds', fname)) as f:
                d = json.load(f)
            return [x.get(key, x.get(alt, 0)) for x in d['folds']]
        folds = {
            'LR':  lf('logistic_regression_folds.json'),
            'SVM': lf('svm_folds.json'),
            'RF':  lf('random_forest_folds.json'),
            'ResNet-50':      lf('resnet50_folds.json', key='f1_thr05'),
            'U-Net ResNet34': lf('unet_folds.json',    key='f1_pixel_thr05'),
        }
    except Exception:
        folds = FOLDS_FB
    return df, ch, folds

df, ch, fold_data = load_data()

# Paleta
C = {
    'RedEdge':'#DC2626','Topo':'#F97316','SAR':'#EAB308','Optico':'#3B82F6',
    'red':'#DC2626','gray':'#9CA3AF','dark':'#374151',
    'clasico':'#6B7280','dl':'#C4B5FD','blue_dark':'#1E3A5F','blue_light':'#93C5FD',
}

def mpl_defaults():
    plt.rcParams.update({
        'figure.facecolor':'white','axes.facecolor':'white',
        'axes.edgecolor':'#CCCCCC','axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.color':'#EEEEEE','grid.linewidth':0.8,
        'font.family':'sans-serif','font.size':11,
        'xtick.color':'#555555','ytick.color':'#555555','axes.labelcolor':'#333333',
    })

# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏔️ Landslide ML · Colombia")
    st.markdown("---")
    seccion = st.radio(
        "Navegación",
        ["Inicio", "Exploración", "Argumento", "Conclusión"],
        index=0,
    )
    st.markdown("---")
    st.markdown("**Filtros globales**")
    umbral_f1 = st.slider("Umbral F1 de referencia", 0.60, 0.95, 0.80, 0.01)
    todos_modelos = ['LR','SVM','RF','ResNet-50','EfficientNet','U-Net']
    sel_modelos = st.multiselect(
        "Modelos a mostrar",
        todos_modelos,
        default=todos_modelos,
    )
    mostrar_lit = st.toggle("Mostrar benchmarks literatura", value=True)
    st.markdown("---")
    st.caption("Datos: Landslide4Sense Dataset\nModelos entrenados en Google Colab")

# Datos filtrados
MODELOS_F1 = {
    'LR':0.7886,'SVM':0.7974,'RF':0.8368,
    'ResNet-50':0.7840,'EfficientNet':0.7554,'U-Net':0.4443,
}
MODELOS_TIPO = {
    'LR':'Clásico','SVM':'Clásico','RF':'Clásico',
    'ResNet-50':'DL','EfficientNet':'DL','U-Net':'DL',
}
modelos_vis  = [m for m in todos_modelos if m in sel_modelos]
f1_vis       = [MODELOS_F1[m] for m in modelos_vis]
tipo_vis     = [MODELOS_TIPO[m] for m in modelos_vis]

def color_modelo(nombre):
    if nombre == 'RF': return C['red']
    return C['clasico'] if MODELOS_TIPO.get(nombre) == 'Clásico' else C['gray']

# ══════════════════════════════════════════════════════════════════
# SECCIÓN 1 — INICIO
# ══════════════════════════════════════════════════════════════════
if seccion == "Inicio":
    st.markdown("# Detección de Deslizamientos con Machine Learning")
    st.markdown("### ¿Qué nos dicen los modelos sobre cómo proteger Colombia?")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h2 style="border:none;color:#DC2626;font-size:2rem!important">400–600</h2><p>eventos de deslizamiento por año en Colombia<br><small style="color:#9CA3AF">(SGC, 2023)</small></p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h2 style="border:none;color:#DC2626;font-size:2rem!important">0</h2><p>datasets etiquetados colombianos disponibles públicamente</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h2 style="border:none;color:#DC2626;font-size:2rem!important">14</h2><p>bandas satelitales analizadas · Sentinel-1/2 + DEM</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="argumento-box"><strong>Argumento central</strong><br>"El modelo más sofisticado no siempre gana — y en Colombia, donde no hay un dataset propio, elegir mal el modelo y el protocolo de evaluación puede ser la diferencia entre una herramienta útil y una que falla cuando más se necesita."</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### El recorrido de este análisis")
        for paso, desc in [
            ("1. Exploración","¿Qué muestran los datos sin hipótesis previas?"),
            ("2. Trampa metodológica","El resultado cambia según cómo evalúes el modelo"),
            ("3. Contexto global","¿Dónde estamos frente a la literatura internacional?"),
            ("4. Complejidad","Más parámetros no garantizan mejor detección"),
            ("5. Disponibilidad","¿Qué información satelital necesita Colombia?"),
            ("6. Confianza","¿En qué modelo confiar con datos escasos?"),
        ]:
            st.markdown(f'<div class="hallazgo-box"><strong>{paso}</strong> — {desc}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("#### Modelos evaluados")
        resumen = pd.DataFrame({
            'Modelo': todos_modelos,
            'Tipo':   [MODELOS_TIPO[m] for m in todos_modelos],
            'F1':     [MODELOS_F1[m] for m in todos_modelos],
        }).sort_values('F1', ascending=False).reset_index(drop=True)
        resumen['F1'] = resumen['F1'].map(lambda x: f"{x:.3f}")
        st.dataframe(resumen, use_container_width=True, hide_index=True)
        st.caption("Usa el panel lateral para filtrar modelos y ajustar el umbral F1.")

# ══════════════════════════════════════════════════════════════════
# SECCIÓN 2 — EXPLORACIÓN
# ══════════════════════════════════════════════════════════════════
elif seccion == "Exploración":
    st.markdown("# Fase Exploratoria — ¿Dónde estamos?")
    st.markdown("Cinco hallazgos sin hipótesis previas. Los datos tal como son.")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "H1 · F1 por modelo",
        "H2 · Precisión vs Cobertura",
        "H3 · Canales satelitales",
        "H4 · Brecha de señal",
        "H5 · Variabilidad folds",
    ])

    # ── H1 ───────────────────────────────────────────────────────
    with tab1:
        st.markdown("### ¿Qué tan bien detecta cada modelo un deslizamiento?")
        st.markdown('<div class="hallazgo-box">Métrica: <strong>F1-Score</strong> — equilibrio entre detectar los que sí ocurren y no generar falsas alarmas.</div>', unsafe_allow_html=True)

        if not modelos_vis:
            st.warning("Selecciona al menos un modelo en el panel lateral.")
        else:
            orden = sorted(range(len(modelos_vis)), key=lambda i: f1_vis[i])
            m_ord = [modelos_vis[i] for i in orden]
            f_ord = [f1_vis[i] for i in orden]
            cols_bar = [C['red'] if m == 'RF' else (C['clasico'] if MODELOS_TIPO[m]=='Clásico' else C['gray']) for m in m_ord]

            mpl_defaults()
            fig, ax = plt.subplots(figsize=(9, max(3, len(m_ord)*0.7)))
            bars = ax.barh(m_ord, f_ord, color=cols_bar, height=0.55, zorder=3)
            ax.axvline(umbral_f1, color=C['dark'], lw=1.4, ls='--', zorder=2)
            ax.text(umbral_f1+0.002, len(m_ord)-0.4, f'F1={umbral_f1:.2f}', fontsize=9, color=C['dark'])
            for bar, val in zip(bars, f_ord):
                ax.text(val+0.004, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=10)
            ax.set_xlim(0, 0.95); ax.set_xlabel('F1-Score'); ax.set_ylabel('Modelo')
            ax.set_title('F1-Score por Modelo — Detección de Deslizamientos', fontsize=13)
            ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
            leg = [mpatches.Patch(color=C['gray'], label='Deep Learning'),
                   mpatches.Patch(color=C['clasico'], label='Modelos clásicos'),
                   mpatches.Patch(color=C['red'], label='Mejor resultado')]
            ax.legend(handles=leg, loc='lower right', fontsize=9, frameon=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── H2 ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("### ¿Precisión o cobertura? ¿Podemos tener los dos?")
        st.markdown('<div class="hallazgo-box">En detección de desastres, <strong>fallar en detectar uno real</strong> suele ser más costoso que una falsa alarma.</div>', unsafe_allow_html=True)

        datos_pr = {'LR':(0.7971,0.7806),'SVM':(0.8193,0.7777),'RF':(0.7439,0.9569),'ResNet-50':(0.7219,0.8771)}
        lit_pr = [
            ('Ghorbanzadeh et al. (2022)', 0.717, '#FECACA'),
            ('Lv et al. — L4S (2022)',     0.739, '#F87171'),
            ('Liu et al. (2024)',           0.760, '#EF4444'),
            ('Enhanced U-Net++ (2025)',     0.841, '#B91C1C'),
        ]
        pr_vis = {k:v for k,v in datos_pr.items() if k in sel_modelos}

        mpl_defaults()
        fig, ax = plt.subplots(figsize=(7.5, 6))
        r_arr = np.linspace(0.62, 0.999, 400)
        if mostrar_lit:
            for nombre, f1v, cl in lit_pr:
                p_arr = f1v * r_arr / (2*r_arr - f1v)
                mask = (p_arr > 0) & (p_arr <= 1.0)
                ax.plot(r_arr[mask], p_arr[mask], color=cl, lw=1.2, ls='--', zorder=1)
                vr, vp = r_arr[mask], p_arr[mask]
                if len(vr):
                    ax.text(vr[0]+0.003, vp[0]+0.005, nombre, fontsize=7.5, color=cl, ha='left', va='bottom', clip_on=True)
        for nm, (prec, rec) in pr_vis.items():
            ax.scatter(rec, prec, s=120, color=color_modelo(nm), zorder=5, edgecolors='white', lw=1.5)
            off = {'LR':(-0.014,0.012),'SVM':(0.005,0.012),'RF':(0.005,-0.022),'ResNet-50':(0.005,0.012)}
            dx,dy = off.get(nm,(0.005,0.012))
            ax.text(rec+dx, prec+dy, nm, fontsize=9.5, color=color_modelo(nm), fontweight='bold')
        ax.set_xlim(0.62,1.02); ax.set_ylim(0.62,0.90)
        ax.set_xlabel('Cobertura (Recall)'); ax.set_ylabel('Precisión')
        ax.set_title('Precisión vs Cobertura por Modelo', fontsize=13)
        if mostrar_lit:
            ax.plot([],[],color='#EF4444',ls='--',lw=1,label='Benchmarks literatura (ISO-F1)')
            ax.legend(fontsize=8.5, frameon=False, loc='upper left')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── H3 ───────────────────────────────────────────────────────
    with tab3:
        st.markdown("### ¿Qué información satelital separa mejor las clases?")
        st.markdown('<div class="hallazgo-box">Δ = diferencia entre el valor medio de reflectancia en zonas <strong>con</strong> y <strong>sin</strong> deslizamiento.</div>', unsafe_allow_html=True)

        canales = [
            ('S2-B7 RedEdge3',0.8073,'RedEdge'),('S2-B6 RedEdge2',0.5625,'RedEdge'),
            ('ALOS DEM',0.1954,'Topo'),('S1-VH SAR',0.1882,'SAR'),
            ('DEM Slope',0.0430,'Topo'),('S2-B8A NIR-A',0.0221,'Optico'),
        ]
        mpl_defaults()
        fig, ax = plt.subplots(figsize=(9, 4))
        UMBRAL = 0.12
        bars = ax.barh([c[0] for c in canales],[c[1] for c in canales],
                       color=[C[c[2]] for c in canales], height=0.55, zorder=3)
        for bar, (nm, val, grp) in zip(bars, canales):
            bw = bar.get_width()
            y_mid = bar.get_y()+bar.get_height()/2
            if bw >= UMBRAL:
                ax.text(bw+0.008, y_mid, f'{val:.3f}', va='center', fontsize=10)
            else:
                ax.text(bw+0.018, y_mid, f'{val:.3f}', va='center', fontsize=9.5,
                        bbox=dict(boxstyle='round,pad=0.2',facecolor='#F9FAFB',edgecolor=C[grp],alpha=0.9,linewidth=0.8))
        ax.set_xlim(0,0.95); ax.set_xlabel('Brecha de señal (Δ)'); ax.set_ylabel('Canal satelital')
        ax.set_title('Canales Satelitales más Discriminativos', fontsize=13)
        leg = [mpatches.Patch(color=C['RedEdge'],label='Banda RedEdge (S2)'),
               mpatches.Patch(color=C['Topo'],label='Topografía'),
               mpatches.Patch(color=C['SAR'],label='Radar SAR (S1)'),
               mpatches.Patch(color=C['Optico'],label='Óptico NIR')]
        ax.legend(handles=leg,loc='upper right',fontsize=9,frameon=True,framealpha=0.95,edgecolor='#E5E7EB')
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── H4 ───────────────────────────────────────────────────────
    with tab4:
        st.markdown("### ¿Cómo se diferencian las clases en los canales clave?")
        st.markdown('<div class="hallazgo-box">Cada canal muestra el valor medio de reflectancia en zonas <strong>con</strong> deslizamiento (sólido) y <strong>sin</strong> él (hueco). La distancia entre ambos es la brecha Δ.</div>', unsafe_allow_html=True)

        dot_data = [
            ('S2-B7 RedEdge3',2.0209,1.2136,'RedEdge'),('S2-B6 RedEdge2',1.4782,0.9157,'RedEdge'),
            ('ALOS DEM',1.2739,1.0786,'Topo'),('S1-VH SAR',1.2488,1.0606,'SAR'),
            ('S2-B8A NIR-A',1.0397,1.0176,'Optico'),('DEM Slope',1.0703,1.0274,'Topo'),
        ]
        mpl_defaults()
        fig, ax = plt.subplots(figsize=(9,4.5))
        for i,(nm,pos,neg,grp) in enumerate(dot_data):
            col = C[grp]
            ax.plot([neg,pos],[i,i],color='#D1D5DB',lw=1.8,zorder=2)
            ax.scatter(pos,i,s=90,color=col,zorder=4)
            ax.scatter(neg,i,s=90,color='white',zorder=4,edgecolors=col,linewidths=1.8)
            ax.text(max(pos,neg)+0.04,i,f'Δ={pos-neg:.2f}',va='center',fontsize=9,color='#6B7280')
        ax.set_yticks(range(len(dot_data))); ax.set_yticklabels([d[0] for d in dot_data])
        ax.set_xlim(0.8,2.5); ax.set_xlabel('Reflectancia media normalizada'); ax.set_ylabel('Canal satelital')
        ax.set_title('Brecha de Señal entre Clases — Top 6 Canales\n(sólido = con deslizamiento · hueco = sin deslizamiento)', fontsize=11)
        leg = [mpatches.Patch(color=C['RedEdge'],label='RedEdge (S2)'),
               mpatches.Patch(color=C['Topo'],label='Topografía'),
               mpatches.Patch(color=C['SAR'],label='SAR (S1)'),
               mpatches.Patch(color=C['Optico'],label='Óptico NIR'),
               Line2D([0],[0],marker='o',color='w',markerfacecolor='#555',markersize=9,label='Con deslizamiento'),
               Line2D([0],[0],marker='o',color='w',markerfacecolor='white',markeredgecolor='#555',markeredgewidth=1.5,markersize=9,label='Sin deslizamiento')]
        ax.legend(handles=leg,loc='upper right',fontsize=8.5,frameon=True,framealpha=0.6,edgecolor='#E5E7EB',ncol=2)
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    # ── H5 ───────────────────────────────────────────────────────
    with tab5:
        st.markdown("### ¿El rendimiento es estable entre experimentos?")
        st.markdown('<div class="hallazgo-box">Cada punto es el resultado en uno de los 5 subconjuntos de prueba. Un modelo inconsistente puede fallar en condiciones no vistas.</div>', unsafe_allow_html=True)

        folds_vis = {k:v for k,v in fold_data.items() if any(k.startswith(m.split()[0]) for m in sel_modelos)}
        if not folds_vis:
            folds_vis = fold_data

        medias = {k:np.mean(v) for k,v in folds_vis.items()}
        orden_f = sorted(folds_vis.keys(), key=lambda k: medias[k], reverse=True)

        mpl_defaults()
        fig, ax = plt.subplots(figsize=(9, max(4, len(orden_f)*0.9)))
        col_f = [C['red'] if 'RF' in m else '#9CA3AF' for m in orden_f]
        bp = ax.boxplot([folds_vis[m] for m in orden_f], vert=False, patch_artist=True,
                        widths=0.45, showfliers=False,
                        medianprops=dict(color='white',lw=2),
                        whiskerprops=dict(color='#9CA3AF'),
                        capprops=dict(color='#9CA3AF'),
                        boxprops=dict(linewidth=0))
        for patch, col in zip(bp['boxes'], col_f):
            patch.set_facecolor(col); patch.set_alpha(0.75)
        rng = np.random.default_rng(42)
        for i,(m,col) in enumerate(zip(orden_f,col_f),start=1):
            vals = folds_vis[m]
            jitter = rng.uniform(-0.12,0.12,len(vals))
            ax.scatter(vals,[i+j for j in jitter],color=col,s=50,zorder=5,alpha=0.9)
            ax.text(np.mean(vals)+0.003,i+0.28,f'x̄={np.mean(vals):.3f}  Std={np.std(vals,ddof=1):.3f}',fontsize=8.5,color=col,va='bottom')
        ax.axvline(umbral_f1,color=C['dark'],lw=1.2,ls='--',zorder=2)
        ax.text(umbral_f1+0.001,0.55,f'F1={umbral_f1:.2f}',fontsize=8.5,color=C['dark'])
        ax.set_yticks(range(1,len(orden_f)+1)); ax.set_yticklabels(orden_f)
        ax.set_xlabel('F1-Score por fold'); ax.set_ylabel('Modelo')
        ax.set_title('Consistencia entre Experimentos — 5 Folds por Modelo', fontsize=13)
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

# ══════════════════════════════════════════════════════════════════
# SECCIÓN 3 — ARGUMENTO
# ══════════════════════════════════════════════════════════════════
elif seccion == "Argumento":
    st.markdown("# Fase Aclaratoria — Del hallazgo al argumento")
    st.markdown("Las mismas preguntas, ahora respondidas con contexto colombiano.")
    st.markdown('<div class="argumento-box">Problema Colombia → ¿qué modelo responde?</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab_a1, tab_a2, tab_a3, tab_a4, tab_a5 = st.tabs([
        "A1 · La trampa metodológica",
        "A2 · Colombia vs mundo",
        "A3 · Complejidad ≠ rendimiento",
        "A4 · Qué necesita Colombia",
        "A5 · Confianza con pocos datos",
    ])

    # ── A1 ───────────────────────────────────────────────────────
    with tab_a1:
        st.markdown("### El resultado cambia según cómo evalúes el modelo")
        col_txt, col_fig = st.columns([1, 2])
        with col_txt:
            st.markdown('<div class="argumento-box">Si alguien en Colombia implementa un modelo y solo reporta el protocolo más favorable, puede <strong>sobrestimar el rendimiento real</strong> en condiciones locales.</div>', unsafe_allow_html=True)
            st.markdown("**Dos protocolos:**")
            st.markdown("- **14 bandas** del satélite (optimizado, n=1500)")
            st.markdown("- **Características básicas** del terreno (comparable literatura, n=3799)")
        with col_fig:
            f1_opt  = {'LR':0.7886,'SVM':0.7974,'RF':0.8368}
            f1_base = {'LR':0.7512,'SVM':0.7340,'RF':0.7891}
            f1_dl   = {'ResNet-50':0.7840,'EfficientNet':0.7554,'U-Net':0.4443}
            m_cl = [m for m in ['LR','SVM','RF'] if m in sel_modelos]
            m_dl = [m for m in ['ResNet-50','EfficientNet','U-Net'] if m in sel_modelos]

            mpl_defaults()
            fig, ax = plt.subplots(figsize=(9,5))
            x = np.arange(len(m_cl)); ancho = 0.32
            x_dl = np.arange(len(m_cl), len(m_cl)+len(m_dl))
            if m_dl:
                ax.bar(x_dl,[f1_dl[m] for m in m_dl],width=ancho*1.9,color='#E5E7EB',zorder=3)
                for xi,m in zip(x_dl,m_dl):
                    ax.text(xi,f1_dl[m]+0.008,f'{f1_dl[m]:.3f}',ha='center',fontsize=9.5,color='#9CA3AF')
            if m_cl:
                b1=ax.bar(x-ancho/2,[f1_opt[m] for m in m_cl],width=ancho,color=C['blue_dark'],zorder=4)
                b2=ax.bar(x+ancho/2,[f1_base[m] for m in m_cl],width=ancho,color=C['blue_light'],zorder=4)
                for bar,m in zip(b1,m_cl):
                    ax.text(bar.get_x()+bar.get_width()/2,f1_opt[m]+0.008,f'{f1_opt[m]:.3f}',ha='center',fontsize=9.5,color=C['blue_dark'],fontweight='bold')
                for bar,m in zip(b2,m_cl):
                    ax.text(bar.get_x()+bar.get_width()/2,f1_base[m]+0.008,f'{f1_base[m]:.3f}',ha='center',fontsize=9.5,color=C['dark'])
                if 'RF' in m_cl:
                    ri = m_cl.index('RF')
                    ax.annotate('',xy=(ri+ancho/2,f1_base['RF']+0.004),xytext=(ri-ancho/2,f1_opt['RF']+0.004),
                                arrowprops=dict(arrowstyle='<->',color=C['red'],lw=2))
                    ax.text(ri+0.04,(f1_opt['RF']+f1_base['RF'])/2+0.03,f'Δ={f1_opt["RF"]-f1_base["RF"]:.3f}',fontsize=9,color=C['red'],fontweight='bold')
            ax.axhline(umbral_f1,color=C['dark'],lw=1.2,ls='--',zorder=2)
            ax.set_xticks(list(x)+list(x_dl)); ax.set_xticklabels(m_cl+m_dl)
            sep_x = len(m_cl)-0.5
            ax.axvline(sep_x,color='#E5E7EB',lw=1.5)
            if m_cl: ax.text(len(m_cl)/2-0.2,0.93,'Modelos clásicos',fontsize=8.5,color='#9CA3AF',ha='center')
            if m_dl: ax.text(len(m_cl)+len(m_dl)/2-0.3,0.93,'Deep Learning',fontsize=8.5,color='#9CA3AF',ha='center')
            ax.set_ylim(0.35,0.97); ax.set_ylabel('F1-Score')
            ax.set_title('El protocolo de evaluación cambia el resultado',fontsize=11,pad=12)
            leg=[mpatches.Patch(color=C['blue_dark'],label='14 bandas del satélite'),
                 mpatches.Patch(color=C['blue_light'],label='Características básicas del terreno'),
                 mpatches.Patch(color='#E5E7EB',label='Deep Learning')]
            ax.legend(handles=leg,loc='upper right',fontsize=9,frameon=True,framealpha=0.95,edgecolor='#E5E7EB')
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    # ── A2 ───────────────────────────────────────────────────────
    with tab_a2:
        st.markdown("### ¿Dónde estamos frente a la literatura internacional?")
        col_txt, col_fig = st.columns([1, 2])
        with col_txt:
            st.markdown('<div class="argumento-box">Colombia registra 400–600 deslizamientos por año. Los modelos de referencia se entrenaron en Nepal, Perú e Italia — terrenos con <strong>regímenes de lluvia y vegetación distintos</strong> a los Andes colombianos.</div>', unsafe_allow_html=True)
            st.markdown("**Implicación:** Sin un dataset colombiano, cualquier modelo es una extrapolación.")
        with col_fig:
            datos_pr_a2={'LR':(0.7971,0.7806),'SVM':(0.8193,0.7777),'RF':(0.7439,0.9569),'ResNet-50':(0.7219,0.8771)}
            lit_a2=[('Ghorbanzadeh et al. (2022)',0.717,'#FECACA'),('Lv et al. — L4S (2022)',0.739,'#F87171'),
                    ('Liu et al. (2024)',0.760,'#EF4444'),('Enhanced U-Net++ (2025)',0.841,'#B91C1C')]
            pr_vis_a2={k:v for k,v in datos_pr_a2.items() if k in sel_modelos}
            mpl_defaults()
            fig,ax=plt.subplots(figsize=(7.5,6))
            r_arr=np.linspace(0.62,0.999,400)
            if mostrar_lit:
                for nombre,f1v,cl in lit_a2:
                    p_arr=f1v*r_arr/(2*r_arr-f1v); mask=(p_arr>0)&(p_arr<=1.0)
                    ax.plot(r_arr[mask],p_arr[mask],color=cl,lw=1.3,ls='--',zorder=1)
                    vr,vp=r_arr[mask],p_arr[mask]
                    if len(vr): ax.text(vr[0]+0.003,vp[0]+0.005,nombre,fontsize=7.5,color=cl,ha='left',va='bottom',clip_on=True)
            for nm,(prec,rec) in pr_vis_a2.items():
                mk='D' if MODELOS_TIPO.get(nm)=='DL' else 'o'
                ax.scatter(rec,prec,s=130,color=color_modelo(nm),zorder=6,edgecolors='white',lw=1.5,marker=mk)
                off={'LR':(-0.016,0.013),'SVM':(0.006,0.013),'RF':(0.006,-0.024),'ResNet-50':(0.006,0.013)}
                dx,dy=off.get(nm,(0.006,0.013))
                ax.text(rec+dx,prec+dy,nm,fontsize=9.5,color=color_modelo(nm),fontweight='bold')
            ax.axhspan(0.58,0.645,alpha=0.07,color='#FCD34D',zorder=1)
            ax.text(0.635,0.592,'Sin datos colombianos — zona de incertidumbre',fontsize=7.5,color='#92400E',style='italic')
            ax.set_xlim(0.62,1.02); ax.set_ylim(0.58,0.90)
            ax.set_xlabel('Cobertura (Recall)'); ax.set_ylabel('Precisión')
            ax.set_title('Comparación con benchmarks internacionales',fontsize=11)
            leg_a2=[Line2D([0],[0],marker='o',color='w',markerfacecolor=C['clasico'],markersize=10,label='Modelos clásicos'),
                    Line2D([0],[0],marker='D',color='w',markerfacecolor=C['dark'],markersize=9,label='Deep Learning'),
                    Line2D([0],[0],marker='o',color='w',markerfacecolor=C['red'],markersize=10,label='RF — mejor F1')]
            if mostrar_lit: leg_a2.append(Line2D([0],[0],color='#EF4444',ls='--',lw=1.5,label='Literatura (ISO-F1)'))
            ax.legend(handles=leg_a2,loc='upper left',fontsize=8.5,frameon=True,framealpha=0.95,edgecolor='#E5E7EB')
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    # ── A3 ───────────────────────────────────────────────────────
    with tab_a3:
        st.markdown("### Complejidad del modelo vs. resultado real")
        col_txt, col_fig = st.columns([1, 2])
        with col_txt:
            st.markdown('<div class="argumento-box"><strong>La objeción más común:</strong> "Los modelos profundos siempre son mejores."<br><br>Los datos muestran lo contrario. U-Net — la arquitectura más compleja — tiene el <strong>peor F1</strong>.</div>', unsafe_allow_html=True)
            st.markdown("**¿Por qué?** Tamaño del dataset, disponibilidad de bandas, y características del terreno.")
        with col_fig:
            m_a3=[m for m in ['U-Net','EfficientNet','ResNet-50','LR','SVM','RF'] if m in sel_modelos]
            f_a3=[MODELOS_F1[m] for m in m_a3]
            comp={'LR':'Baja','SVM':'Media','RF':'Media','ResNet-50':'Alta','EfficientNet':'Alta','U-Net':'Alta'}
            col_a3=[C['red'] if m=='RF' else (C['dl'] if MODELOS_TIPO[m]=='DL' else C['clasico']) for m in m_a3]
            mpl_defaults()
            fig,ax=plt.subplots(figsize=(9,max(3.5,len(m_a3)*0.75)))
            bars=ax.barh(m_a3,f_a3,color=col_a3,height=0.55,zorder=3)
            for bar,m in zip(bars,m_a3):
                ax.text(min(bar.get_width()*0.05,0.04),bar.get_y()+bar.get_height()/2,
                        f'Complejidad: {comp[m]}',va='center',fontsize=8.5,color='white',fontweight='bold')
            for bar,val in zip(bars,f_a3):
                ax.text(val+0.005,bar.get_y()+bar.get_height()/2,f'{val:.3f}',va='center',fontsize=10)
            ax.axvline(umbral_f1,color=C['dark'],lw=1.2,ls='--',zorder=2)
            ax.text(umbral_f1+0.002,len(m_a3)-0.4,f'F1={umbral_f1:.2f}',fontsize=8.5,color=C['dark'])
            ax.set_xlim(0,0.97); ax.set_xlabel('F1-Score'); ax.set_ylabel('Modelo')
            ax.set_title('Complejidad del modelo vs. resultado real\nMás parámetros no garantizan mejor detección',fontsize=11)
            leg=[mpatches.Patch(color=C['dl'],label='Deep Learning'),
                 mpatches.Patch(color=C['clasico'],label='Modelos clásicos'),
                 mpatches.Patch(color=C['red'],label='Mejor resultado')]
            ax.legend(handles=leg,loc='lower right',fontsize=9,frameon=False)
            ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    # ── A4 ───────────────────────────────────────────────────────
    with tab_a4:
        st.markdown("### ¿Qué información satelital necesita Colombia?")
        col_txt, col_fig = st.columns([1, 2])
        with col_txt:
            st.markdown('<div class="argumento-box">Las bandas <strong>RedEdge</strong> son las más discriminativas — y están disponibles en Sentinel-2, con cobertura nacional gratuita.<br><br>El desafío no es el satélite: <strong>es tener imágenes etiquetadas post-evento.</strong></div>', unsafe_allow_html=True)
        with col_fig:
            canales_a4=[
                ('S2-B7 RedEdge3',0.8073,'RedEdge',True),('S2-B6 RedEdge2',0.5625,'RedEdge',True),
                ('ALOS DEM',0.1954,'Topo',False),('S1-VH SAR',0.1882,'SAR',True),
                ('DEM Slope',0.0430,'Topo',False),('S2-B8A NIR-A',0.0221,'Optico',True),
            ]
            UMBRAL2=0.12
            mpl_defaults()
            fig,ax=plt.subplots(figsize=(9.5,4.2))
            bars=ax.barh([c[0] for c in canales_a4],[c[1] for c in canales_a4],
                         color=[C[c[2]] for c in canales_a4],height=0.55,zorder=3)
            for bar,(nm,val,grp,disp) in zip(bars,canales_a4):
                bw=bar.get_width(); y_mid=bar.get_y()+bar.get_height()/2
                estado='Copernicus' if disp else 'DEM externo'
                col_e='#16A34A' if disp else '#B45309'
                if bw>=UMBRAL2:
                    ax.text(0.008,y_mid,estado,va='center',fontsize=8,color='white',fontweight='bold')
                    ax.text(bw+0.012,y_mid,f'{val:.3f}',va='center',fontsize=10)
                else:
                    ax.text(bw+0.055,y_mid,estado,va='center',fontsize=7.5,color=col_e,fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2',facecolor='#F9FAFB',edgecolor=col_e,alpha=0.9,linewidth=0.8))
                    ax.text(bw+0.18,y_mid,f'{val:.3f}',va='center',fontsize=9.5)
            ax.set_xlim(0,0.97); ax.set_xlabel('Brecha de señal (Δ)'); ax.set_ylabel('Canal satelital')
            ax.set_title('Canales más discriminativos y su disponibilidad para Colombia',fontsize=11)
            leg=[mpatches.Patch(color=C['RedEdge'],label='Banda RedEdge (S2)'),
                 mpatches.Patch(color=C['Topo'],label='Topografía'),
                 mpatches.Patch(color=C['SAR'],label='Radar SAR (S1)'),
                 mpatches.Patch(color=C['Optico'],label='Óptico NIR')]
            ax.legend(handles=leg,loc='upper right',fontsize=8.5,frameon=True,framealpha=0.95,edgecolor='#E5E7EB')
            ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

    # ── A5 ───────────────────────────────────────────────────────
    with tab_a5:
        st.markdown("### ¿En cuál modelo confiar con datos escasos?")
        col_txt, col_fig = st.columns([1, 2])
        with col_txt:
            st.markdown('<div class="argumento-box">En Colombia, los datos serán escasos y de un solo evento. La <strong>consistencia entre folds</strong> es el indicador más importante — no la media más alta.</div>', unsafe_allow_html=True)
            st.markdown("**RF:** Std=0.008 — el más consistente")
            st.markdown("**SVM:** Std=0.033 — mayor variabilidad")
        with col_fig:
            folds_a5={k:v for k,v in fold_data.items() if any(k.startswith(m.split()[0]) for m in sel_modelos)} or fold_data
            medias_a5={k:np.mean(v) for k,v in folds_a5.items()}
            stds_a5={k:np.std(v,ddof=1) for k,v in folds_a5.items()}
            orden_a5=sorted(folds_a5.keys(),key=lambda k:medias_a5[k],reverse=True)
            col_a5=[C['red'] if 'RF' in m else '#9CA3AF' for m in orden_a5]
            mpl_defaults()
            fig,ax=plt.subplots(figsize=(9,max(4,len(orden_a5)*0.9)))
            bp=ax.boxplot([folds_a5[m] for m in orden_a5],vert=False,patch_artist=True,
                          widths=0.45,showfliers=False,
                          medianprops=dict(color='white',lw=2),
                          whiskerprops=dict(color='#9CA3AF'),capprops=dict(color='#9CA3AF'),
                          boxprops=dict(linewidth=0))
            for patch,col in zip(bp['boxes'],col_a5):
                patch.set_facecolor(col); patch.set_alpha(0.75)
            rng=np.random.default_rng(42)
            for i,(m,col) in enumerate(zip(orden_a5,col_a5),start=1):
                vals=folds_a5[m]; jitter=rng.uniform(-0.12,0.12,len(vals))
                ax.scatter(vals,[i+j for j in jitter],color=col,s=50,zorder=5,alpha=0.9)
                ax.text(np.mean(vals)+0.003,i+0.28,f'x̄={medias_a5[m]:.3f}  Std={stds_a5[m]:.3f}',fontsize=8.5,color=col,va='bottom')
            ax.axvline(umbral_f1,color=C['dark'],lw=1.2,ls='--',zorder=2)
            ax.text(umbral_f1+0.001,0.55,f'F1={umbral_f1:.2f}',fontsize=8.5,color=C['dark'])
            rf_idx=next((i for i,m in enumerate(orden_a5) if 'RF' in m),None)
            if rf_idx is not None:
                rf_pos=rf_idx+1
                ax.annotate('Dispersión más pequeña\n→ más confiable con datos nuevos',
                            xy=(np.mean(folds_a5[orden_a5[rf_idx]]),rf_pos),
                            xytext=(0.695,rf_pos+1.4),
                            arrowprops=dict(arrowstyle='->',color=C['red'],lw=1.5),
                            fontsize=8.5,color=C['red'],
                            bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor=C['red'],alpha=0.9))
            ax.set_yticks(range(1,len(orden_a5)+1)); ax.set_yticklabels(orden_a5)
            ax.set_xlabel('F1-Score por fold'); ax.set_ylabel('Modelo')
            ax.set_title('Consistencia del modelo como indicador de confianza',fontsize=13)
            ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
            plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

# ══════════════════════════════════════════════════════════════════
# SECCIÓN 4 — CONCLUSIÓN
# ══════════════════════════════════════════════════════════════════
elif seccion == "Conclusión":
    st.markdown("# Conclusión — Lo que necesita Colombia")
    st.markdown('<div class="argumento-box" style="font-size:1.05rem">La arquitectura viene después del contexto. Random Forest — interpretable y eficiente con pocos datos — es un punto de partida más sólido que redes profundas diseñadas para miles de imágenes segmentadas.</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Síntesis de hallazgos")
        tabla = pd.DataFrame({
            'Hallazgo': ['F1 por modelo','Precisión vs Cobertura','Canales satelitales',
                         'Brecha de señal','Variabilidad folds'],
            'Observación': [
                'RF supera F1=0.80; U-Net queda muy por debajo',
                'RF prioriza cobertura (Recall=0.96)',
                'Bandas RedEdge dominan la discriminación',
                'RedEdge3 tiene brecha 4× mayor que SAR-VH',
                'RF es el más consistente (Std=0.008)',
            ],
        })
        st.dataframe(tabla, use_container_width=True, hide_index=True)

        st.markdown("### Tabla comparativa completa")
        df_show = df.copy()
        df_show['F1 medio'] = df_show['F1 medio'].map(lambda x: f'{x:.4f}' if pd.notna(x) else '—')
        df_show['Recall']   = df_show['Recall'].map(lambda x: f'{x:.4f}' if pd.notna(x) else '—')
        df_show['Precisión']= df_show['Precisión'].map(lambda x: f'{x:.4f}' if pd.notna(x) else '—')
        st.dataframe(df_show[['Modelo','Tipo','F1 medio','Precisión','Recall']], use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Condiciones para Colombia")
        condiciones = [
            ("Dataset etiquetado nacional", "No existe", "#FEE2E2", "#DC2626",
             "Sin esto, cualquier modelo es una extrapolación"),
            ("Bandas satelitales disponibles", "Sentinel-2 disponible (RedEdge incluido)", "#DCFCE7", "#16A34A",
             "La señal está — faltan etiquetas post-evento"),
            ("Protocolo de evaluación honesto", "Depende del estudio", "#FEF9C3", "#CA8A04",
             "Usar 5 folds con protocolo comparable a literatura"),
            ("Arquitectura apropiada", "RF como punto de partida", "#DCFCE7", "#16A34A",
             "Interpretable, eficiente con pocos datos, robusto"),
        ]
        for cond, estado, bg, color, implicacion in condiciones:
            st.markdown(f"""
            <div style="background:{bg};border-left:4px solid {color};padding:10px 14px;
                        border-radius:4px;margin-bottom:10px;">
                <strong style="color:{color}">{cond}</strong><br>
                <span style="color:#374151">Estado: {estado}</span><br>
                <small style="color:#6B7280">{implicacion}</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Llamado a acción")
        st.markdown("""
        Replicar este análisis en Colombia requiere tres pasos concretos:

        1. **Etiquetar imágenes** de eventos históricos en el SGC (Servicio Geológico Colombiano)
        2. **Descargar bandas RedEdge** de Sentinel-2 para las zonas afectadas (API Copernicus)
        3. **Entrenar y evaluar** con 5 folds y protocolo comparable a literatura internacional

        La arquitectura correcta emerge del contexto — no al revés.
        """)

    st.markdown("---")
    st.caption("Datos: Landslide4Sense Dataset · Modelos entrenados en Google Colab · "
               "Benchmarks: Ghorbanzadeh (2022), L4S Competition (2022), Liu et al. (2024), Enhanced U-Net++ (2025)")
