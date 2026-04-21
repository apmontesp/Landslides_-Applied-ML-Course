# Transferibilidad a Colombia — Contexto Andino

**Proyecto:** Landslide4Sense — Análisis de Dominio  
**Fecha:** 2026

---

## 1. Motivación

El dataset Landslide4Sense cubre cuatro regiones geográficas (Hokkaido, Lombardia, Nepal, Pakistan), con características geomorfológicas distintas al contexto andino colombiano. Este análisis evalúa la **brecha de dominio** (domain gap) entre el dataset de entrenamiento y las condiciones de los Andes colombianos, y propone estrategias de adaptación.

---

## 2. Diferencias de Dominio

### 2.1 Cobertura de Nubes

| Región | Nubosidad promedio |
|--------|-------------------|
| Hokkaido (Japón) | 20–30% |
| Lombardia (Italia) | 25–35% |
| Nepal | 30–40% |
| **Andes colombianos** | **60–80%** |

La alta nubosidad en Colombia es la principal limitación para el uso directo de imágenes ópticas Sentinel-2. Los canales SAR (Sentinel-1, ALOS) son **insensibles a las nubes** y adquieren mayor importancia relativa en el contexto colombiano.

**Implicación:** Los modelos entrenados con los 14 canales deberán adaptar la ponderación de canales. Una estrategia de *channel dropout* durante el entrenamiento (enmascarar aleatoriamente canales ópticos) podría mejorar la robustez ante ausencia de información óptica.

### 2.2 Tipo de Vegetación

| Característica | Landslide4Sense | Andes colombianos |
|---------------|----------------|------------------|
| Cobertura vegetal | Bosques templados, estepa | Bosques húmedos tropicales |
| Índice NDVI típico | 0.3–0.7 | 0.7–0.9 |
| Densidad del dosel | Media | Alta |

La mayor densidad de vegetación en los Andes colombianos puede obscurecer señales de deslizamientos recientes en canales ópticos. Sin embargo, las bandas **Red-Edge (B5, B6, B7)** son sensibles al estrés vegetal incluso en doseles densos, lo que refuerza su importancia para el contexto colombiano.

### 2.3 Características Litológicas

| Región | Litología dominante |
|--------|-------------------|
| Hokkaido | Volcánica (depósitos piroclásticos) |
| Lombardia | Carbonatos y pizarras alpinas |
| Nepal | Metamórfica e intrusiva |
| Pakistan | Sedimentaria árida |
| **Andes colombianos** | **Volcánica-metamórfica (Formación Barroso, Complejo Cajamarca)** |

La litología volcánica de los Andes colombianos es parcialmente comparable a la región de Hokkaido, lo cual sugiere que los modelos podrían transferir mejor en zonas volcánicas andinas (Galeras, Nevado del Ruiz, Chiles-Cerro Negro).

### 2.4 Régimen de Lluvia y Gatilladores

| Región | Gatillador principal |
|--------|---------------------|
| Hokkaido | Sismos |
| Lombardia | Lluvias intensas |
| Nepal | Monzón |
| Pakistan | Sismos |
| **Andes colombianos** | **Lluvias + Sismos + Vulcanismo** |

Colombia tiene gatilladores múltiples simultáneos. Los deslizamientos por lluvia (más frecuentes) son similares a los de Lombardia, lo que sugiere que ese subconjunto del dataset puede ser el más transferible.

---

## 3. Zonas de Alto Riesgo en Colombia

Las siguientes zonas del departamento de Antioquia presentan alta susceptibilidad y potencial para validación/fine-tuning:

| Municipio | Coordenadas aprox. | Eventos históricos | Tipo |
|-----------|--------------------|--------------------|------|
| Abriaquí | 6.6°N, 76.1°W | Múltiples (2009, 2019) | Lluvia, sísmica |
| Dabeiba | 7.0°N, 76.3°W | Frecuente | Lluvia |
| Salgar | 5.9°N, 75.9°W | 2015 (>100 víctimas) | Lluvia extrema |
| Ituango | 7.2°N, 75.8°W | 2018 (presa) | Mixto |

Estos municipios tienen:
- Cobertura histórica de Sentinel-1 y Sentinel-2
- Informes de inventario de movimientos en masa del SGC (Servicio Geológico Colombiano)
- Condiciones morfológicas representativas de los Andes húmedos

---

## 4. Estimación Cuantitativa del Domain Gap

Para cuantificar la brecha de dominio entre Landslide4Sense y datos colombianos, se propone el uso de la **distancia máxima de discrepancia media (MDD)** en el espacio de features del encoder:

```python
# Pseudo-código para estimación del domain gap
encoder = build_resnet50(pretrained=True).features   # Extractor de features
f_source = extract_features(encoder, landslide4sense_train)
f_target = extract_features(encoder, colombia_patches)

# Distancia en espacio de features
mmd_dist = maximum_mean_discrepancy(f_source, f_target, kernel="rbf")
print(f"Domain gap (MMD): {mmd_dist:.4f}")
# Interpretación: < 0.1 → bajo gap, > 0.5 → alto gap
```

---

## 5. Estrategias de Adaptación

### 5.1 Fine-tuning con datos colombianos (recomendada)

Con **50–200 parches anotados** de Antioquia, se puede realizar fine-tuning del modelo preentrenado en Landslide4Sense:

```bash
# Ejemplo de fine-tuning local
python scripts/run_training.py \
    --config configs/efficientnet_b4.yaml \
    --data_root ./data_colombia \
    --output_dir ./results/colombia_finetune \
    --epochs 20 \
    --lr_backbone 0.000001   # LR muy bajo para no olvidar el conocimiento previo
```

### 5.2 Domain Adaptation sin etiquetas

Si no hay datos anotados de Colombia, se pueden aplicar técnicas de **adaptación de dominio no supervisada**:

- **CORAL (Correlation Alignment):** alinea las estadísticas de segundo orden entre dominios
- **Adversarial Domain Adaptation:** discriminador que hace que los features sean agnósticos al dominio
- **CycleGAN:** traducción de imagen fuente → target para aumentar el dataset

### 5.3 Channel Dropout para robustez ante nubes

Durante el entrenamiento en Landslide4Sense, aplicar *channel dropout* sobre los canales ópticos (0–6, 11–13) con probabilidad 0.3:

```python
if random.random() < 0.3:
    n_drop = random.randint(1, 4)
    drop_idx = random.sample([0,1,2,3,4,5,6,11,12,13], n_drop)
    patch[:, :, drop_idx] = 0.0   # Simular canal cubierto por nubes
```

### 5.4 Análisis de inventarios SGC

El Servicio Geológico Colombiano (SGC) mantiene el **Inventario Nacional de Movimientos en Masa (SIMMA)**, que documenta más de 35,000 eventos con coordenadas y fechas. Este inventario puede usarse para:

1. Seleccionar parches de Sentinel-2/Sentinel-1 colombianos con deslizamientos confirmados
2. Generar máscaras de segmentación por buffer alrededor del punto de reporte
3. Validar los modelos sobre eventos conocidos

---

## 6. Plan de Trabajo Futuro

| Fase | Actividad | Recursos necesarios |
|------|-----------|---------------------|
| 1 | Adquisición de imágenes Sentinel-1/2 sobre Antioquia (2020–2025) | Google Earth Engine (gratuito) |
| 2 | Consulta SIMMA para eventos con fecha y coordenadas | SGC (datos abiertos) |
| 3 | Anotación manual de 100 parches positivos | 20–30 horas/persona |
| 4 | Fine-tuning del mejor modelo (EfficientNet-B4 FT) | GPU × 2 épocas |
| 5 | Evaluación cuantitativa del domain gap (MMD) | — |
| 6 | Publicación en repositorio con datos colombianos | — |

---

## 7. Conclusión

Los modelos entrenados sobre Landslide4Sense son un **punto de partida sólido** para la detección en Colombia, pero no son directamente aplicables sin adaptación por las diferencias en nubosidad (>60%), vegetación tropical y régimen de lluvia. La estrategia recomendada es:

1. **Corto plazo:** Usar EfficientNet-B4 fine-tuned para alertas sobre imágenes SAR (canales 7–10, robustos ante nubes).
2. **Mediano plazo:** Recolectar 100–200 parches colombianos anotados y hacer fine-tuning local.
3. **Largo plazo:** Desarrollar un sistema de detección automática post-evento integrado con Sentinel Hub y el SIMMA del SGC.

Esta investigación sienta las bases metodológicas y computacionales para ese desarrollo.

---

**Referencias clave:**

- SGC — SIMMA: [simma.sgc.gov.co](https://simma.sgc.gov.co)
- Sentinel Hub: [sentinel-hub.com](https://www.sentinel-hub.com)
- Google Earth Engine: [earthengine.google.com](https://earthengine.google.com)
- Ghorbanzadeh et al. (2022). Landslide4Sense. *IEEE TGRS*, 60, 1–17.
