# Transferibilidad a Colombia — Contexto Andino

**Proyecto:** Landslide4Sense — Análisis de Brechas de Dominio
**Fecha:** 2026

---

## 1. Motivación

El dataset Landslide4Sense cubre regiones de Asia, Europa y América del Sur con características geomorfológicas distintas al contexto andino colombiano. Wang y Brenning [15] cuantifican que la transferencia de modelos entre contextos geomorfológicos sin adaptación puede producir degradaciones de hasta el 25% en métricas de desempeño. Aplicado al mejor resultado obtenido (RF F1=0.837), esto proyecta **F1≈0.63** en datos colombianos sin reentrenamiento local. Este análisis evalúa las brechas de dominio concretas e identifica estrategias de adaptación. Enfoques recientes de mapeo cross-domain [19][20] sugieren que la armonización de datos multi-sensor y técnicas de enmascaramiento morfológico pueden reducir estas brechas de forma sistemática.

---

## 2. Brechas de Dominio Identificadas

### 2.1 Cobertura de Nubes

| Región | Nubosidad promedio |
|--------|-------------------|
| Hokkaido (Japón) | 20–30% |
| Lombardia (Italia) | 25–35% |
| Nepal | 30–40% |
| **Andes colombianos** | **60–80%** |

La alta nubosidad en Colombia es la principal limitación para el uso directo de imágenes ópticas Sentinel-2 (bandas B2–B7, B8A, B11, B12). Los canales SAR (Sentinel-1 VV, VH) mantienen penetración atmosférica pero pierden sensibilidad en pendientes pronunciadas por efecto de layover [2].

**Implicación:** Los modelos deben ponderarse hacia los canales SAR y DEM (canales 7–10). Una estrategia de *channel dropout* durante el entrenamiento —enmascarar aleatoriamente canales ópticos con probabilidad 0.3— puede mejorar la robustez ante ausencia de información óptica.

### 2.2 Tipo de Vegetación

| Característica | Landslide4Sense | Andes colombianos |
|---------------|----------------|------------------|
| Cobertura vegetal | Bosques templados, estepa | Bosques húmedos tropicales |
| Índice NDVI típico | 0.3–0.7 | 0.7–0.9 |
| Densidad del dosel | Media | Alta |

La mayor densidad de vegetación puede obscurecer señales de deslizamientos recientes en canales ópticos. La señal RedEdge (Ch12–13), la más discriminativa según el EDA (Δ=+0.807 y +0.563), se satura en coberturas densas, reduciendo su poder de separación entre clases [3].

### 2.3 Características Litológicas

| Región | Litología dominante |
|--------|-------------------|
| Hokkaido | Volcánica (depósitos piroclásticos) |
| Lombardia | Carbonatos y pizarras alpinas |
| Nepal | Metamórfica e intrusiva |
| **Andes colombianos** | **Volcánica-metamórfica (Formación Barroso, Complejo Cajamarca)** |

El ambiente volcánico-metamórfico de los Andes colombianos afecta las firmas espectrales en bandas SWIR y SAR, diferenciándose de las regiones loéssicas o calcáreas predominantes en el dataset [4]. La litología volcánica es parcialmente comparable a Hokkaido, lo que sugiere que los modelos podrían transferir mejor en zonas volcánicas andinas (Galeras, Nevado del Ruiz).

### 2.4 Régimen de Lluvia y Gatilladores

| Región | Gatillador principal |
|--------|---------------------|
| Hokkaido | Sismos |
| Lombardia | Lluvias intensas |
| Nepal | Monzón |
| **Andes colombianos** | **Lluvias + Sismos + Vulcanismo** |

Colombia tiene gatilladores múltiples simultáneos. Los deslizamientos por lluvia (más frecuentes) son similares a los de Lombardia, lo que sugiere que ese subconjunto del dataset puede ser el más transferible [5].

---

## 3. Cuantificación de la Brecha

Wang y Brenning [15] demuestran que la transferencia sin adaptación de dominio puede producir degradaciones de F1 de hasta 25 puntos porcentuales. Enfoques de mapeo cross-domain como los propuestos por Yu et al. [19] —armonización de datasets heterogéneos multisensor— y Chen et al. [20] —enmascaramiento morfológico para extracción robusta— ofrecen rutas concretas para reducir esta brecha sin requerir grandes inventarios anotados localmente.

| Escenario | F1 proyectado |
|-----------|-------------|
| RF en Landslide4Sense (resultado real) | 0.837 |
| RF transferido a Colombia sin adaptación | ~0.63 |
| RF con fine-tuning local (proyectado) | 0.75–0.82 |

---

## 4. Zonas de Alta Susceptibilidad en Colombia

Las siguientes zonas del departamento de Antioquia son prioritarias para validación y colecta de datos:

| Municipio | Coordenadas aprox. | Eventos históricos | Tipo de gatillador |
|-----------|--------------------|--------------------|-------------------|
| Abriaquí | 6.6°N, 76.1°W | Múltiples (2009, 2019) | Lluvia, sísmica |
| Dabeiba | 7.0°N, 76.3°W | Frecuente | Lluvia |
| Salgar | 5.9°N, 75.9°W | 2015 (>100 víctimas) | Lluvia extrema |
| Ituango | 7.2°N, 75.8°W | 2018 (presa) | Mixto |

Estas zonas tienen cobertura histórica de Sentinel-1 y Sentinel-2, informes de inventario del SGC [4] y condiciones morfológicas representativas de los Andes húmedos.

---

## 5. Estrategias de Adaptación

### 5.1 Fine-tuning con datos colombianos (recomendada)

Con **50–200 parches anotados** de Antioquia, se puede realizar fine-tuning del modelo preentrenado en Landslide4Sense:

```bash
python scripts/run_training.py \
    --config configs/resnet50.yaml \
    --data_root ./data_colombia \
    --output_dir ./results/colombia_finetune \
    --epochs 20 \
    --lr_backbone 0.000001
```

### 5.2 Domain Adaptation (CORAL) sin etiquetas

Si no hay datos anotados de Colombia, se pueden aplicar técnicas de adaptación de dominio no supervisada:

- **CORAL (Correlation Alignment):** alinea las estadísticas de segundo orden entre dominios fuente y destino
- **Cross-domain mapping [19][20]:** armonización de datasets heterogéneos para reducir la brecha espectral
- **Channel Dropout:** enmascarar canales ópticos aleatoriamente durante entrenamiento para simular alta nubosidad

```python
# Channel dropout para robustez ante nubes
if random.random() < 0.3:
    n_drop = random.randint(1, 4)
    drop_idx = random.sample([0,1,2,3,4,5,6,11,12,13], n_drop)
    patch[:, :, drop_idx] = 0.0   # Simula canal cubierto por nubes
```

### 5.3 Uso del inventario SGC/SIMMA

El Servicio Geológico Colombiano (SGC) mantiene el **Inventario Nacional de Movimientos en Masa (SIMMA)**, que documenta más de 35,000 eventos con coordenadas y fechas. Este inventario puede usarse para:

1. Seleccionar parches de Sentinel-1/2 colombianos con deslizamientos confirmados
2. Generar máscaras de segmentación por buffer alrededor del punto de reporte
3. Validar los modelos sobre eventos históricos conocidos

---

## 6. Plan de Trabajo Futuro

| Fase | Actividad | Recursos |
|------|-----------|---------|
| 1 | Adquisición de imágenes Sentinel-1/2 sobre Antioquia (2020–2025) | Google Earth Engine (gratuito) |
| 2 | Consulta SIMMA para eventos con fecha y coordenadas | SGC (datos abiertos) |
| 3 | Anotación de 100–200 parches positivos con DJI Mini 4 Pro [6] | Campo + 20–30 horas/persona |
| 4 | Fine-tuning del RF y U-Net con datos mixtos + CORAL | GPU × 2 épocas |
| 5 | Evaluación cuantitativa del domain gap (MMD) | — |
| 6 | Publicación con datos colombianos en repositorio | — |

---

## 7. Conclusión

Los modelos entrenados sobre Landslide4Sense son un **punto de partida sólido** para la detección en Colombia, pero no son directamente aplicables sin adaptación por las diferencias en nubosidad (>60%), vegetación tropical y litología volcánica-metamórfica. La estrategia recomendada es:

1. **Corto plazo:** Usar RF fine-tuned para alertas sobre imágenes SAR (canales 7–10, robustos ante nubes).
2. **Mediano plazo:** Recolectar 100–200 parches colombianos anotados con UAV y hacer fine-tuning local [6].
3. **Largo plazo:** Desarrollar un sistema de detección automática post-evento integrado con Sentinel Hub y el SIMMA del SGC, incorporando técnicas de domain adaptation cross-domain [19][20].

---

**Referencias:**
[2] Ge et al. (2023) · [3] Thirugnanam et al. (2022) · [4] SGC (2017) · [5] Ayala-García & Ospino-Ramos (2019) · [6] Sun et al. (2024) · [15] Wang & Brenning (2023) · [19] Yu et al. (2025) · [20] Chen et al. (2025)

- SGC — SIMMA: [simma.sgc.gov.co](https://simma.sgc.gov.co)
- Google Earth Engine: [earthengine.google.com](https://earthengine.google.com)
- Sentinel Hub: [sentinel-hub.com](https://www.sentinel-hub.com)
