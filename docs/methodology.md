# Metodología — Detección de Deslizamientos con Deep Learning

**Proyecto:** Landslide4Sense — Evaluación Comparativa de Arquitecturas CNN  
**Fecha:** 2026

---

## 1. Diseño Experimental

El experimento sigue un protocolo de **5-Fold Stratified Cross-Validation** sobre las 3,799 imágenes de TrainData, con estratificación por etiqueta de parche (presencia/ausencia de deslizamiento). Se usan 4 folds para entrenamiento y 1 para validación, rotando el fold de validación en cada iteración. Las métricas reportadas son la media ± desviación estándar sobre los 5 folds.

### Pregunta de investigación

> ¿En qué medida el fine-tuning de arquitecturas CNN sobre el dataset multi-espectral Landslide4Sense supera a los baselines clásicos y al entrenamiento desde cero, y qué implicaciones tiene para la transferibilidad a contextos geomorfológicos andinos colombianos?

### Hipótesis

1. El fine-tuning supera al entrenamiento desde cero en al menos 0.08 puntos de F1.
2. La fusión de canales SAR y ópticos mejora la detección respecto a solo RGB.
3. U-Net produce mejores mapas de probabilidad pixel-level que las redes de clasificación.

---

## 2. Dataset

**Landslide4Sense** (ISPRS 2022): parches de 128×128 píxeles con 14 canales espectrales capturados de múltiples sensores sobre 4 regiones geográficas (Hokkaido, Lombardia, Nepal, Pakistan).

### Canales de entrada

| Índice | Fuente | Banda | Resolución |
|--------|--------|-------|-----------|
| 0–6 | Sentinel-2 | B2, B3, B4, B8, B8A, B11, B12 | 10–20 m |
| 7–8 | Sentinel-1 SAR | VV, VH | 10 m |
| 9–10 | ALOS PALSAR | DEM, Pendiente | 12.5 m |
| 11–13 | Sentinel-2 Red-Edge | B5, B6, B7 | 20 m |

### Hallazgos EDA que impactan el diseño

- **Balance de clases real:** 58.7% positivos / 41.3% negativos → `pos_weight = 0.703`
- **Small object detection:** área mediana = 2.04% del parche → justifica Dice Loss y U-Net
- **Canales más discriminativos:** RedEdge3 (Δ=+0.807), RedEdge2 (Δ=+0.563), DEM (Δ=+0.195)

---

## 3. Preprocesamiento

### Normalización

Normalización **Z-score por canal** sobre los 14 canales de forma independiente:

```
patch_norm[c] = (patch[c] - μ[c]) / (σ[c] + ε)
```

donde μ[c] y σ[c] se estiman sobre una muestra representativa de TrainData.

### Data Augmentation

Aplicada **solo en entrenamiento**, de forma consistente entre imagen y máscara:

| Transformación | Probabilidad | Nota |
|---------------|-------------|------|
| Flip horizontal | 0.50 | Equivariante a orientación |
| Flip vertical | 0.50 | |
| Rotación 90°/180°/270° | 0.50 | k aleatorio en {1,2,3} |
| Perturbación de brillo | 0.30 | Solo canales ópticos 0–6, ×U[0.8,1.2] |
| Ruido gaussiano | 0.20 | σ=0.02, todos los canales |

---

## 4. Adaptación a 14 Canales (Mean-Initialization)

Los pesos de la primera capa convolucional de ImageNet (3 canales) se adaptan a 14 canales mediante **replicación cíclica con escalado**:

```python
new_w[:, :14, :, :] = old_w.repeat(1, 5, 1, 1)[:, :14, :, :] * (3/14)
```

Esta estrategia, documentada como superior a la inicialización aleatoria en dominios multiespectrales, preserva la norma de activación esperada del feature map de salida.

---

## 5. Protocolo de Fine-Tuning

### Fase 1 — Adaptación (épocas 1–5)

- Backbone preentrenado **congelado** (sin gradientes)
- Solo la cabeza de clasificación se actualiza
- Objetivo: adaptar la cabeza al espacio de features 14-canal

### Fase 2 — Fine-Tuning completo (épocas 6–N)

- Backbone **descongelado** completamente
- **Learning rate diferencial:**
  - Cabeza: `lr_head = 1e-4`
  - Backbone: `lr_backbone = 1e-5` (10× menor)
- Scheduler: `CosineAnnealingLR(T_max=50, η_min=1e-7)`

### Hiperparámetros comunes

| Parámetro | Valor |
|-----------|-------|
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Épocas máximas | 50 (ResNet, EfficientNet) / 60 (U-Net) |
| Patience (early stopping) | 15 épocas |
| Métrica monitoreada | val_F1 (clasificadores) / val_IoU (U-Net) |
| Semilla | 42 |

---

## 6. Funciones de Pérdida

### Clasificadores (ResNet-50, EfficientNet-B4)

**Weighted Binary Cross-Entropy:**

```
L_BCE = -(w₁·y·log(ŷ) + (1−y)·log(1−ŷ))
```

con `w₁ = pos_weight = 0.703 = n_neg / n_pos`.

### Segmentación (U-Net + ResNet-34)

**Dice + BCE combinadas (50/50):**

```
L = 0.5 · L_BCE + 0.5 · L_Dice

L_Dice = 1 - (2·|P∩T| + ε) / (|P| + |T| + ε)
```

La componente Dice es crítica para deslizamientos de pequeña escala (área mediana 2.04%), ya que no colapsa ante el desbalance espacial extremo dentro del parche.

---

## 7. Métricas de Evaluación

| Métrica | Tarea | Fórmula |
|---------|-------|---------|
| F1-score | Clasificación de parche | 2·P·R / (P+R) |
| AUC-ROC | Clasificación de parche | Área bajo curva ROC |
| Precisión | Clasificación de parche | TP / (TP+FP) |
| Recall | Clasificación de parche | TP / (TP+FN) |
| IoU (Jaccard) | Segmentación pixel | TP / (TP+FP+FN) |
| Tiempo de inferencia | Eficiencia | ms/imagen en GPU |

### Umbral de decisión

Se optimiza post-entrenamiento sobre la curva Precisión-Recall del fold de validación, seleccionando el umbral que maximiza F1. El umbral de referencia es 0.5; el umbral óptimo típicamente cae entre 0.35–0.55.

---

## 8. Baseline Random Forest

Como punto de comparación clásico, se implementa un clasificador Random Forest (200 árboles, `max_depth=None`) sobre **características HOG** extraídas del canal RGB:

- Tamaño de celda: 16×16 píxeles
- Bloques: 2×2 celdas
- Orientaciones: 9

El RF no utiliza los canales SAR, DEM ni Red-Edge; esto cuantifica la contribución del espectro extendido en los modelos deep learning.

---

## 9. Reproducibilidad

Todas las semillas están fijadas con `set_seed(42)`:
- `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`
- `cudnn.deterministic = True`, `cudnn.benchmark = False`

Los archivos de configuración YAML en `configs/` documentan todos los hiperparámetros. Los checkpoints del mejor modelo por fold se guardan en `results/<modelo>/fold_<k>/best_model.pth`.
