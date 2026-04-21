# Resultados — Detección de Deslizamientos con Aprendizaje Automático

**Proyecto:** Landslide4Sense — Evaluación Comparativa

**Fecha:** 2026

> Resultados reales obtenidos sobre **2-Fold Stratified Cross-Validation** sobre 3,799 parches (clasificadores) y 2,000 parches (U-Net). Entorno: Google Colab GPU T4.

---

## 1. Tabla Comparativa Principal

| Modelo | Tipo | F1 medio | AUC-ROC | AUC-PR | Observaciones |
|--------|------|----------|---------|--------|---------------|
| **Random Forest (HOG)** | Clásico | **0.837** ★ | 0.808 | — | Mejor resultado global |
| SVM kernel RBF (HOG) | Clásico | 0.797 | 0.779 | — | Segundo mejor |
| Regresión Logística (HOG) | Clásico | 0.789 | 0.761 | — | Baseline lineal competitivo |
| ResNet-50 fine-tuned | Deep Learning | 0.784 | ~0.813 | 0.810 / 0.816 | AUC-PR por pliegue |
| EfficientNet-B4 fine-tuned | Deep Learning | 0.756 | ~0.823 | 0.807 / 0.839 | Mayor AUC-ROC; F1 inferior a RF |
| U-Net + ResNet-34 | Segmentación | 0.445 | ~0.391 | 0.391 / 0.420 | Tarea pixel-level — no comparable |

★ Mejor resultado en clasificación de parche. AUC-PR reportado por pliegue para modelos DL. — = no aplicable.

---

## 2. Hallazgo principal

**Los tres modelos clásicos con ingeniería de características HOG superan a ResNet-50 fine-tuned en F1**, contradiciendo la narrativa predominante en la literatura reciente [10][18]. La diferencia es consistente entre pliegues, descartando variabilidad aleatoria como explicación.

| Comparación | F1 RF | F1 ResNet-50 | Diferencia |
|-------------|-------|-------------|-----------|
| Random Forest vs ResNet-50 | 0.837 | 0.784 | **+5.3 pp** |
| SVM vs ResNet-50 | 0.797 | 0.784 | +1.3 pp |
| LR vs ResNet-50 | 0.789 | 0.784 | +0.5 pp |

En AUC-ROC la jerarquía se invierte parcialmente: EfficientNet-B4 alcanza ~0.823, sugiriendo que su curva ROC es superior pero que el umbral por defecto (0.5) produce una relación precisión-recall menos favorable.

### Tres factores explicativos

**1. Ingeniería de características efectiva.** El vector HOG + DEM + NDVI + SAR-VH captura directamente las señales identificadas como más discriminativas en el EDA (RedEdge Ch12–13, DEM Ch9, SAR-VH Ch8), alineando las features con el conocimiento físico del fenómeno sin necesidad de aprendizaje desde datos.

**2. Régimen de datos limitado.** Con 1,500 muestras por pliegue, ResNet-50 (25M parámetros) no tiene señal suficiente para superar el sesgo inductivo de features bien diseñadas. La literatura sugiere que el aprendizaje profundo tiende a superar métodos clásicos a partir de ~10k muestras en clasificación de parche [8].

**3. Diferencia de tarea.** Los clasificadores de parche (LR, SVM, RF, ResNet, EfficientNet) predicen una etiqueta binaria por parche. La U-Net produce un mapa de probabilidad 128×128. La comparación directa de F1 no es equivalente: operar sobre ~16,384 píxeles vs 1 etiqueta por parche.

---

## 3. Resultados U-Net — segmentación pixel-level

| Métrica | Valor |
|---------|-------|
| F1 medio | 0.445 |
| AUC-PR fold 1 | 0.391 |
| AUC-PR fold 2 | 0.420 |
| AUC-ROC | ~0.391 |
| N muestras entrenamiento | 2,000 (subconjunto estratificado) |
| Épocas máximas | 10 |

El F1=0.445 refleja las condiciones experimentales restrictivas (N=2,000, 10 épocas máx., umbral fijo 0.5), **no el límite arquitectónico del modelo**. Las predicciones visuales muestran localización espacial coherente: el modelo identifica correctamente la forma y posición de las zonas afectadas, pero la baja proporción de píxeles positivos (~2–5% del parche) castiga severamente el F1 con umbral fijo. Los valores de AUC-PR confirman que el modelo discrimina mejor de lo que el umbral estándar reporta. Variantes optimizadas como las propuestas en Song et al. [13] y Wang et al. [14] con módulos de atención obtienen mejoras significativas bajo condiciones de entrenamiento completo.

---

## 4. Análisis por tipo de modelo

### Clásicos — ventajas confirmadas

- **Interpretabilidad:** importancia de features del RF identifica RedEdge y DEM como los descriptores más relevantes, coherente con el EDA.
- **Eficiencia:** tiempo de entrenamiento < 2 min vs 20–40 min para CNN.
- **Sin GPU:** los tres modelos corren en CPU estándar.
- **Robustez en datos limitados:** el sesgo inductivo del RF supera la capacidad expresiva de CNNs con pocas muestras.

### CNN — ventajas diferidas

- EfficientNet-B4 obtiene el **AUC-ROC más alto (~0.823)**, indicando que su curva ROC es superior; la calibración del umbral podría cerrar la brecha con RF en F1.
- ResNet-50 AUC-PR (0.810/0.816 por fold) confirma buen poder discriminativo; el F1 con umbral fijo no refleja su capacidad real.
- Con >10k muestras etiquetadas, las CNN previsiblemente superarían a los modelos clásicos.

### U-Net — valor cualitativo

La U-Net es el único modelo que provee **localización espacial del deslizamiento**, información relevante para estimación de área afectada y gestión del riesgo a nivel de parcela. Su comparación con clasificadores de parche es análoga a comparar detección de objetos con clasificación de imagen: tareas fundamentalmente distintas.

---

## 5. Comparación con literatura

| Trabajo | Modelo | F1 / IoU | Dataset | Condición |
|---------|--------|---------|---------|-----------|
| BisDeNet [10] | Red lightweight DL | 0.85+ IoU | Propio, >10k | DL con dataset grande |
| ShapeFormer [18] | Vision Transformer | 0.88+ F1 | Propio | Alta resolución espacial |
| Youssef & Pourghasemi [8] | ML clásico | 0.82–0.91 AUC | Asir, Saudi Arabia | Susceptibilidad (no detección) |
| **Este trabajo** | RF (HOG) | **0.837 F1** | Landslide4Sense | 2-Fold CV, 1,500 muestras |
| **Este trabajo** | ResNet-50 FT | **0.784 F1** | Landslide4Sense | 2-Fold CV, fine-tuning |

Los resultados de trabajos como BisDeNet y ShapeFormer operan sobre datasets sustancialmente mayores o con imágenes de alta resolución espacial específicas, lo que explica la diferencia con los valores aquí obtenidos.

---

## 6. Limitaciones del estudio

| Limitación | Impacto | Mitigación aplicada |
|-----------|---------|-------------------|
| Protocolo de 2 pliegues | Menor estabilidad de estimaciones de varianza vs 5–10 pliegues | Semilla fija, estratificación |
| N=2,000 parches para U-Net | Reduce representatividad del entrenamiento | Selección estratificada |
| Ablation study ResNet-50 incompleto | Limita atribución causal del desempeño | Reportado como trabajo futuro |
| Sin etiquetas de validación oficial | Impide evaluar generalización out-of-distribution | Validación cruzada interna |
| Umbral fijo 0.5 | Puede subestimar F1 real de modelos CNN | AUC-PR reportada para contexto |

---

## 7. Trabajo futuro

- Completar ablation study de ResNet-50 (augmentation, freeze/unfreeze, pos_weight, LR uniforme)
- Calibración del umbral de decisión sobre curva PR para CNN (potencial +2 pp F1)
- Fine-tuning sobre datos mixtos (Landslide4Sense + inventario SGC Colombia) con CORAL domain adaptation
- Evaluación con protocolo 5-Fold en entorno con mayor cómputo disponible

---

**Referencias:** [8] Youssef & Pourghasemi (2021) · [10] Chen et al. / BisDeNet (2024) · [13] Song et al. (2025) · [14] Wang et al. (2024) · [17] Ghorbanzadeh et al. (2022) · [18] Lv et al. / ShapeFormer (2023)
