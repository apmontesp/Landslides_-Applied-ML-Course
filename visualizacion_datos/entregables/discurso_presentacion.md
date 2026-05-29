# Discurso de presentación
## "Detección de Deslizamientos con Machine Learning en Colombia"
**Duración estimada: 8–9 minutos** · Ritmo: 1 diapositiva cada ~50–70 segundos

---

## DIAPOSITIVA 1 — Portada
*(~40 seg)*

Buenos días a todos. Voy a presentarles un trabajo sobre detección de deslizamientos
de tierra con Machine Learning en Colombia.
La pregunta central que guía todo el análisis es una muy práctica:
**¿qué modelo elegir cuando no existen datos propios?**
Porque ese es exactamente el escenario en el que estamos.

---

## DIAPOSITIVA 2 — El problema: Colombia y los deslizamientos
*(~70 seg)*

Colombia registra entre 400 y 600 eventos de deslizamiento al año.
Es uno de los países con mayor exposición a este riesgo en América Latina.
Sin embargo —y este es el punto de partida de todo el trabajo— **no existe ningún
dataset etiquetado a nivel nacional** que permita entrenar un modelo supervisado
directamente con datos colombianos.

Lo que sí tenemos es señal satelital: 14 bandas de Sentinel-1, Sentinel-2 y modelos
digitales de elevación, disponibles gratuitamente en Copernicus.

Entonces la pregunta no es "¿tenemos datos?" sino "¿cómo usamos los datos que
existen en la literatura internacional para tomar decisiones en Colombia?"

---

## DIAPOSITIVA 3 — La trampa del protocolo de evaluación
*(~70 seg)*

El primer hallazgo es incómodo: **el mismo modelo puede reportar un F1 muy
diferente dependiendo de cómo se evalúa**.

Esta gráfica muestra que si usamos validación aleatoria —el método más frecuente
en publicaciones— el F1 aparece inflado. Si en cambio usamos validación
geoespacial, que respeta la autocorrelación del terreno, el resultado cae.

El problema no es el modelo: es el protocolo.
Esto significa que comparar papers sin revisar cómo evaluaron es, básicamente,
comparar cosas que no son comparables.
La decisión metodológica que tomamos fue usar 5-fold con separación geoespacial,
coherente con los estándares más rigurosos de la literatura.

---

## DIAPOSITIVA 4 — Colombia vs. la literatura internacional
*(~70 seg)*

Aquí ubicamos los benchmarks disponibles en el espacio Precisión / Recall.
Los puntos de referencia son modelos entrenados en Nepal, Perú e Italia —los países
con datasets etiquetados públicos.

Las líneas punteadas son curvas ISO-F1: cada curva une todos los puntos con el
mismo valor de F1. Sirven como regla de distancia.

Lo que vemos es que los mejores modelos de la literatura se acercan a F1 = 0.85.
Colombia, en el escenario de transferencia de dominio —es decir, aplicando esos
modelos a nuestro territorio sin reentrenamiento— queda entre F1 = 0.72 y 0.78.
**Hay brecha, pero no es abismal.** Con el canal correcto esa brecha puede reducirse.

---

## DIAPOSITIVA 5 — Complejidad del modelo vs. resultado real
*(~60 seg)*

Una intuición común en Machine Learning es que más parámetros implica mejor
rendimiento. Esta gráfica la contradice.

Ordenamos los modelos por número de parámetros —de izquierda a derecha, de
más simple a más complejo— y graficamos su F1 en el protocolo geoespacial.

Random Forest, con sólo unos pocos miles de parámetros, iguala o supera a redes
convolucionales que requieren millones de parámetros y órdenes de magnitud más
tiempo de cómputo.

Para Colombia, con recursos limitados, esto es una buena noticia.

---

## DIAPOSITIVA 6 — ¿Qué información satelital necesita Colombia?
*(~60 seg)*

No todas las bandas aportan igual. Aquí vemos el ranking de importancia relativa
para discriminar píxeles de deslizamiento vs. no-deslizamiento.

Las bandas RedEdge de Sentinel-2 —B5, B6, B7— aparecen consistentemente entre
las más informativas. Son sensibles a cambios en la cubierta vegetal que preceden
o acompañan un deslizamiento.

La buena noticia: estas bandas están disponibles gratuitamente en Copernicus.
No se necesita comprar imágenes de alta resolución para obtener resultados útiles.

---

## DIAPOSITIVA 7 — ¿En cuál modelo confiar con datos escasos?
*(~60 seg)*

Cuando los datos son pocos, un modelo puede tener buen promedio pero alta
varianza entre experimentos. Eso es peligroso: significa que el resultado depende
de la partición, no del modelo.

Este gráfico combina un boxplot por modelo con los puntos individuales de cada fold.
Random Forest tiene la **caja más estrecha** —poca varianza— y la **mediana más alta**.
Eso es exactamente lo que queremos: un modelo que no sólo es bueno en promedio,
sino que es predecible.

---

## DIAPOSITIVA 8 — Comparativa visual: F1 sin y con énfasis de protocolo
*(~40 seg — slide de reflexión metodológica)*

Antes de las conclusiones, quiero hacer una pausa para mostrar las decisiones
visuales del trabajo.

A la izquierda: la primera versión exploratoria de la gráfica de F1. Todos los modelos
en el mismo color, sin distinción por protocolo. Es correcta, pero no comunica nada.

A la derecha: la versión final. Se separaron los protocolos en paneles y se usó
color rojo para marcar el protocolo que infla artificialmente el F1.
**El lector identifica la trampa sin necesidad de leer el eje.**

---

## DIAPOSITIVA 9 — Comparativa visual: Precisión/Recall sin y con curvas ISO-F1
*(~35 seg)*

Lo mismo ocurre con la gráfica de Precisión vs. Recall.

Sin las curvas ISO-F1, los puntos flotan en el espacio sin referencia.
Con las curvas, aparece inmediatamente la distancia al estado del arte internacional.
Una línea de referencia convierte un scatter informativo en un argumento visual.

---

## DIAPOSITIVA 10 — Comparativa visual: boxplot sin y con destaque de RF
*(~35 seg)*

Y en la gráfica de varianza por folds: a la izquierda, todos los modelos con el mismo
peso visual —no hay lectura posible. A la derecha, Random Forest en rojo con su
mediana anotada. El ojo va directo al modelo que nos interesa.

**La visualización no debe mostrar datos; debe apoyar una conclusión.**

---

## DIAPOSITIVA 11 — Conclusión
*(~70 seg)*

Entonces, ¿qué necesita Colombia para tener una herramienta de detección útil?

Primero, lo que no tenemos: un dataset etiquetado nacional. Sin él, cualquier
modelo que construyamos es una transferencia desde otro país.

Lo que sí tenemos: señal Sentinel-2 con bandas RedEdge, disponible gratuitamente.

Lo que necesitamos cuidar: el protocolo de evaluación. Si usamos validación
aleatoria vamos a creer que nuestros modelos son mejores de lo que son.

Y la arquitectura: **Random Forest**. No es el modelo más impresionante, pero es
interpretable, eficiente con pocos datos, y —según este análisis— el más consistente
en el escenario colombiano.

El mensaje final es simple: **la complejidad no es virtud**. Un modelo sencillo que
funciona de forma confiable vale más que una red profunda que falla cuando más se
necesita.

Gracias.

---

*Notas de presentación: hablar despacio en las diapositivas 3 y 4, que son las más
densas conceptualmente. Las diapositivas 8–10 pueden presentarse con transición rápida
si el tiempo apremia — son de apoyo visual, no de contenido nuevo.*
