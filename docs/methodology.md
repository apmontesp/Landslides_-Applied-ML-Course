# Metodología — Detección de Deslizamientos con Aprendizaje Automático

**Proyecto:** Landslide4Sense — Evaluación Comparativa de Modelos Clásicos y CNN

**Fecha:** 2026

---

## 1. Diseño Experimental

El experimento sigue un protocolo de **2-Fold Stratified Cross-Validation** sobre las 3,799 imágenes de TrainData, con estratificación por etiqueta de parche (presencia/ausencia de deslizamiento). Se utilizan 2 pliegues en lugar de los 5 convencionales por restricciones de tiempo de cómputo en el entorno Google Colab con GPU T4, que limita el número de entrenamientos completos de modelos CNN. La semilla aleatoria fija `random_state=42` garantiza reproducibilidad en todos los modelos.

### Pregunta de investigación

> **¿Qué enfoque de aprendizaje automático ofrece el mejor desempeño para la detección automática de deslizamientos de tierra en imágenes multisensoriales de satélite bajo condiciones de datos limitados?**

### Objetivos específicos

- **OE1 —** Caracterizar la distribución espectral, el balance de clases y la calidad del dataset Landslide4Sense mediante análisis exploratorio de datos (EDA), como fundamento para las decisiones de diseño experimental.
- **OE2 —** Evaluar el desempeño de tres clasificadores clásicos (Regresión Logística, SVM y Random Forest) con representación HOG, reportando F1-score y AUC-ROC bajo 2-Fold Stratified CV.
- **OE3 —** Evaluar el desempeño de dos arquitecturas CNN de clasificación (ResNet-50 y EfficientNet-B4) y una de segmentación semántica (U-Net+ResNet-34), reportando F1-score/AUC-ROC y Dice/IoU bajo el mismo protocolo de validación cruzada.
- **OE4 —** Analizar las brechas de dominio espectral, cobertura vegetal y contexto geológico que condicionan la transferibilidad de los modelos evaluados al contexto andino colombiano, determinando estrategias de adaptación potenciales.

### Hipótesis de trabajo

1. El enfoque que mejor combine ingeniería de características y capacidad de generalización superará a los demás en términos de F1-score bajo condiciones de datos limitados.
2. La fusión de canales SAR, DEM y Red-Edge mejora la detección respecto al uso exclusivo de canales ópticos.
3. U-Net provee información espacial adicional (localización del deslizamiento) no disponible en clasificadores de parche.

---

## 2. Dataset

**Landslide4Sense** (ISPRS 2022 [17]): parches de 128×128 píxeles con 14 canales espectrales capturados de múltiples sensores sobre regiones de Asia, Europa y América del Sur.

### Canales de entrada

| Índice | Fuente | Banda | Resolución |
|--------|--------|-------|-----------|
| 0–6 | Sentinel-2 | B2, B3, B4, B8, B8A, B11, B12 | 10–20 m |
| 7–8 | Sentinel-1 SAR | VV, VH | 10 m |
| 9–10 | ALOS PALSAR | DEM, Pendiente | 12.5 m |
| 11–13 | Sentinel-2 Red-Edge | B5, B6, B7 | 20 m |

### Hallazgos EDA que impactan el diseño

- **Balance de clases real:** 2,231 positivos (58.7%) / 1,568 negativos (41.3%) → `pos_weight = 0.70`
- **Small object detection:** área mediana = 2.04% del parche → justifica Dice Loss y segmentación pixel-level
- **Canales más discriminativos:** RedEdge3 (Δ=+0.807), RedEdge2 (Δ=+0.563), DEM (Δ=+0.195), SAR-VH (Δ=+0.188)

---

## 3. Preprocesamiento y protocolo de validación

Todo el preprocesamiento se aplica **dentro de cada pliegue** mediante un Pipeline de scikit-learn para garantizar ausencia de data leakage:

```
SimpleImputer(strategy='median') → StandardScaler
```

El Pipeline se ajusta únicamente sobre el conjunto de entrenamiento del pliegue y se aplica al conjunto de validación sin reajuste. Los conjuntos de validación (245 parches) y prueba (800 parches) del dataset oficial no tienen etiquetas disponibles públicamente, por lo que toda la evaluación se realiza sobre el conjunto de entrenamiento mediante validación cruzada.

---

## 4. Ingeniería de características — Modelos clásicos

Los tres modelos clásicos operan sobre un vector de características extraído de cada parche con `N_SAMPLES=1,500` parches seleccionados aleatoriamente (restricción de memoria en extracción HOG):

| Componente | Descripción | Dim. |
|-----------|-------------|------|
| HOG | orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2) sobre RGB (B4-B3-B2) | 1,764 |
| Pendiente DEM | Media del canal 10 | 1 |
| NDVI | Media de (B8−B4)/(B8+B4) | 1 |
| SAR-VH | Media del canal 8 | 1 |

**Vector total:** 1,767 elementos por parche. Alineado con los canales más discriminativos identificados en el EDA.

---

## 5. Modelos clásicos (notebook 03)

### Regresión Logística (LR)

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000))
])
```

Baseline lineal de referencia. Establece el límite inferior esperable para métodos más complejos.

### SVM kernel RBF

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True))
])
```

Captura fronteras de decisión no lineales sin requerir aprendizaje de representaciones desde datos. StandardScaler obligatorio por sensibilidad a la escala.

### Random Forest

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
```

No requiere escalado (invariante a transformaciones monótonas). Proporciona importancia de features interpretable. Ampliamente utilizado en tareas de susceptibilidad y detección de deslizamientos [8][9].

---

## 6. Modelos de aprendizaje profundo — Clasificación de parche (notebooks 04–05)

### Adaptación a 14 canales (Mean-Initialization)

La primera capa convolucional se reemplaza por una nueva de 14 canales de entrada, inicializada promediando los pesos ImageNet de los 3 canales originales y replicando el resultado. Esto preserva el conocimiento aprendido para extracción de bordes y texturas. Trabajos recientes validan esta estrategia para detección multiespectral [11][12].

### Protocolo de congelación progresiva

- **Fase 1:** Backbone congelado, solo se entrena la cabeza de clasificación (`lr_head=1e-4`)
- **Fase 2:** Backbone descongelado con LR diferencial más bajo (`lr_backbone=1e-5`)

### Hiperparámetros comunes

| Parámetro | Valor |
|-----------|-------|
| Optimizer | AdamW |
| Scheduler | OneCycleLR |
| Loss | Weighted BCE (`pos_weight=0.70`) |
| Épocas máximas | 20 por pliegue |
| Semilla | 42 |

### ResNet-50

25M parámetros. Conexiones residuales para gradientes estables en redes profundas.

### EfficientNet-B4

19M parámetros. Escalado compuesto: depth=1.8, width=1.4, resolution=1.3.

---

## 7. U-Net — Segmentación pixel-level (notebook 06)

Encoder ResNet-34 preentrenado en ImageNet via `segmentation-models-pytorch`. Opera sobre un subconjunto estratificado de `N_SUBSET=2,000` parches por restricciones computacionales del entorno Colab. Variantes optimizadas con módulos de atención y doble canal demuestran mejoras adicionales en este tipo de tarea [13][14].

```
Loss = 0.5 × DiceLoss + 0.5 × BCE
```

La componente Dice es crítica para deslizamientos de pequeña escala (área mediana 2.04%), ya que no colapsa ante el desbalance espacial extremo dentro del parche.

| Parámetro | Valor |
|-----------|-------|
| batch_size | 16 |
| lr | 1e-3 |
| Scheduler | OneCycleLR |
| AMP | torch.amp.GradScaler/autocast (~40% reducción tiempo) |
| Early stopping | patience=3 |
| Épocas máximas | 10 |

---

## 8. Métricas de evaluación

| Métrica | Tarea | Descripción |
|---------|-------|-------------|
| **F1-score** | Clasificación de parche | Métrica principal — media armónica de precisión y recall |
| AUC-ROC | Clasificación de parche | Poder discriminativo independiente del umbral |
| AUC-PR | Clasificación de parche (CNN) | Curva Precisión-Recall, reportada por pliegue |
| IoU (Jaccard) | Segmentación pixel | TP / (TP+FP+FN) |
| Dice | Segmentación pixel | 2·TP / (2·TP+FP+FN) |

El F1-score es preferido porque: (i) penaliza simétricamente los falsos positivos y negativos, (ii) es robusto a desequilibrios moderados de clase, y (iii) permite comparación directa con la literatura del dataset [17].

---

## 9. Reproducibilidad

- Semilla fija `random_state=42` en todos los modelos
- Pipeline de scikit-learn aplicado dentro de cada fold (sin leakage)
- Caché de etiquetas en JSON para U-Net (evita releer 3,799 archivos HDF5 por ejecución)
- Configuraciones completas en `configs/*.yaml`

---

**Referencias:** [8] Youssef & Pourghasemi (2021) · [9] Zhou (2024) · [11] Uribe-Ventura (2025) · [12] Song (2025) · [13] Song et al. (2025) · [14] Wang et al. (2024) · [17] Ghorbanzadeh et al. (2022)
