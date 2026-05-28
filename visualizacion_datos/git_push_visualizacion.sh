#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# git_push_visualizacion.sh
# Commit y push de la carpeta visualizacion_datos/ al repositorio
# Ejecutar desde la raíz del proyecto: bash visualizacion_datos/git_push_visualizacion.sh
# ══════════════════════════════════════════════════════════════════════

set -e   # detener si algún comando falla

echo "📁 Verificando ubicación..."
if [ ! -f "requirements.txt" ]; then
  echo "❌ Ejecuta este script desde la raíz del proyecto (Landslide_ML/)"
  exit 1
fi

echo ""
echo "📊 Estado actual del repositorio:"
git status --short visualizacion_datos/

echo ""
echo "➕ Añadiendo visualizacion_datos/ al staging area..."
git add visualizacion_datos/

echo ""
echo "📋 Archivos en staging:"
git diff --cached --name-only

echo ""
echo "💾 Creando commit..."
git commit -m "feat(visualizacion_datos): añadir entregables de la asignatura Visualización de Datos

Fase 1 — Exploración y Hallazgos:
- exploracion_hallazgos.ipynb: pregunta de negocio, 4 visualizaciones
  exploratorias (Seaborn/Matplotlib), hallazgos explícitos sobre
  comparación de modelos y canales espectrales discriminativos

Fase 2 — Análisis Aclaratorio:
- comparativa_visual.ipynb: 3 pares exploratorio vs. aclaratorio
  usando go.Figure(), add_annotation() y update_layout() de Plotly
  con framework de 5 preguntas por gráfica

Fase 3 — Dashboard Interactivo (Streamlit):
- app.py: 5 tabs (Comparación Modelos, Señales Terreno, Análisis
  Avanzado, Transferibilidad, Síntesis Final), filtros dinámicos,
  benchmarking con literatura, predicciones LORO por región,
  curvas de entrenamiento y gap analysis

Datos:
- data/: CSVs y JSONs de resultados copiados del proyecto
- data/figures/: 64 figuras de análisis avanzado, GradCAM,
  SHAP, transferibilidad y síntesis final"

echo ""
echo "🚀 Haciendo push a origin..."
git push origin HEAD

echo ""
echo "✅ Push completado exitosamente."
echo "   Rama actual: $(git branch --show-current)"
echo "   Último commit: $(git log --oneline -1)"
