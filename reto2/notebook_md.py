MD = {
"portada": r"""# Reto 2 — Predicción de accidentalidad vial en El Poblado

**Aprendizaje Automático 2026-2 · Profesor: Pablo Andrés Saldarriaga · Universidad EAFIT**

**Realizado por: Santiago Vélez Casallas**

**Objetivo:** estimar, para cada pareja (barrio, hora) de los 22 barrios de El Poblado, la probabilidad de que ocurra
**al menos un accidente de tránsito**, para que la Secretaría de Movilidad decida dónde y a qué hora ubicar a sus agentes.

**Métrica:** ROC-AUC sobre el periodo de evaluación (2019-08-01 a 2019-12-30, 80.256 parejas con etiqueta oculta).

| Referencia (README) | ROC-AUC |
|---|---|
| Aleatorio | 0.5000 |
| Tasa histórica barrio × hora | 0.7589 |
| Baseline del profesor | 0.7739 |

> ### RESULTADO OFICIAL (leaderboard): ROC-AUC = **0.7838**
> Modelo: **LightGBM global** con barrio + calendario + tendencia. Validación walk-forward en 6 cortes: **0.7845**
> en promedio (sección 8).

**Ruta del notebook (sigue el README punto por punto):** 1. Exploración y calidad · 2. Target y unión de tablas ·
3. Variables · 4. Validación · 5. Modelado · 6. Desbalance y umbral · 7. Métricas más allá del ROC-AUC ·
8. Resultado oficial · 9. Caso de uso y limitaciones.""",

"entorno": r"""**Entorno.** El entorno del curso (`uv sync` con el `pyproject.toml` del repositorio) trae numpy, pandas,
scikit-learn y matplotlib. Este notebook usa además `lightgbm`, `seaborn` e `imbalanced-learn` (SMOTE). La celda
siguiente instala solo las que falten y funciona en Google Colab, en el entorno `uv` y en una instalación normal de
Python. La base de datos (~85 MB) se descarga sola si no está en la carpeta. El notebook completo tarda unos 5 minutos.""",

"s1": r"""## 1. Exploración y calidad de datos

### 1.1 Estructura general""",

"s1_1": r"""En la Tabla 1:

1. `clima` es el **universo**: una fila por barrio y hora de 2017 a 2019 (577.720 filas). `accidentes` y
   `raw_accidentes` llegan solo hasta el **2019-07-31**: después de esa fecha están las etiquetas ocultas.
2. `accidentes` tiene 9.138 parejas (barrio, hora) positivas y `raw_accidentes` 9.428 accidentes individuales. La
   diferencia son horas con más de un accidente en el mismo barrio (sección 2).
3. Los 22 barrios están escritos igual en las tres tablas y no hay parejas duplicadas.
4. `clima` tiene 235.132 celdas faltantes; `raw_accidentes`, 21. Se revisan en 1.4 y 1.5.

### 1.2 Distribución temporal""",

"s1_2": r"""**Lectura de la Figura 1:**

1. **(a) Nivel mensual:** unos 304 accidentes por mes, entre 219 (ene-2018) y 369 (may-2017). No hay una tendencia
   fuerte: 3.858 en 2017 y 3.512 en 2018. Los meses bajos coinciden con vacaciones (enero, mitad de año).
2. **(b) Perfil semanal:** el patrón dominante es la **hora del día**. De lunes a viernes el riesgo sube desde las 6 h
   y tiene su máximo a las **17 h**. El sábado el pico se adelanta al mediodía, y el domingo es bajo y plano.
   Barrio, hora y tipo de día son la base de todo el modelo.

### 1.3 Distribución espacial""",

"s1_3": r"""**Lectura de la Figura 2:**

1. **(a)** El riesgo es muy desigual entre barrios: Villa Carlota tiene 1.187 accidentes y Lalinde 87 (**14 veces**
   menos). Cuatro barrios (Villa Carlota, La Aguacatala, Manila y El Poblado) concentran el 42 % de los accidentes.
2. **(b)** Los accidentes se agrupan sobre las vías principales. El barrio, como variable categórica, resume bien esta
   geografía.
3. **(c)** Hay **5 accidentes de `laflorida`** geocodificados en longitud −75.66, a unos 10 km, en el corregimiento de
   San Antonio de Prado. Es un error de geocodificación (1.5).

### 1.4 Clima""",

"s1_4": r"""**Lectura de la Figura 3 y la Tabla 2:**

1. **Bloque sin lluvia:** `precipIntensity`, `precipProbability` y `summary` faltan en 39.463 filas (6.8 %). Son
   1.801 horas, casi todas entre mayo y agosto de 2018 (Figura 3a), y en cada una falta para los **22 barrios a la
   vez**: es una caída de la fuente, no un hueco por barrio.
2. **`windBearing`** falta en el 12.7 % de las filas, y 57.746 de esas 73.376 tienen `windSpeed = 0`: sin viento no hay
   dirección. No es un error.
3. **El resto de variables** tiene menos del 0.5 % de faltantes.
4. **El clima es prácticamente el mismo en los 22 barrios** (correlación de la temperatura entre barrios a la misma
   hora: 0.997). La Figura 3b muestra que la temperatura sigue un ciclo diario de ~16 °C de madrugada a ~25 °C a
   mediodía, el mismo ciclo que el tráfico.

### 1.5 Inconsistencias entre tablas""",

"s1_5": r"""**Lectura de las Tablas 3 y 4 (qué se encontró y qué se decidió):**

1. **`HORA` en dos formatos:** «11:10 AM» hasta may-2018 y «03:20:00 p. m.» desde feb-2018. Unificados a 24 h, coinciden
   con la hora de `TW` en **todos los registros menos uno** (Tabla 4): el radicado 1679135 dice 01:12 p. m. y quedó
   registrado a la 01:00. Es un error AM/PM.
2. **5 accidentes de `laflorida` en San Antonio de Prado** (Figura 2c). Otros 4 traen el nombre del barrio en `COMUNA`
   pero sus coordenadas están dentro de El Poblado: esos son correctos.
3. **`CLASE`** tiene «Caída Ocupante» (1) y «Caida Ocupante» (376); **`RADICADO`** mezcla «123» y «123.0»; **`DISENO`**
   tiene 21 vacíos. Son problemas de formato que no afectan el target.
4. **`accidentes` y `raw_accidentes` coinciden al 100 %**: los 9.138 pares (barrio, hora) son los mismos en ambas.
5. **Faltan 20 horas completas en `clima`** (440 filas), y en ellas hubo 5 accidentes.

El efecto de corregir 1 y 2 sobre el target se mide en 2.1.

### 1.6 ¿Hace falta imputar el clima?

Antes de imputar hay que saber si el clima aporta. Para cada variable medimos el ROC-AUC que logra sola, de dos
formas: **en bruto** y **dentro de la misma pareja barrio × hora**. Para la segunda se usa la posición del valor
frente a los demás valores de ese barrio a esa misma hora, así que la pregunta es: *a las 17 h en Manila, ¿hay más
accidentes cuando hace más calor?*""",

"s2": r"""**Lectura de la Figura 4 y la Tabla 5:**

1. **En bruto**, la temperatura parece predecir (ROC-AUC 0.671), seguida de la humedad (0.631) y el índice UV (0.618).
2. **Dentro de barrio × hora**, esas mismas variables caen a **0.500**: el azar. La temperatura "predecía" solo porque
   hace calor de día, que es cuando hay tráfico. Esa información ya la da la hora.
3. La lluvia es la única con algo de señal propia (0.53), y es pequeña.

**Decisión:** el modelo final **no usa clima**. En 5.2 se confirma que agregarlo baja el ROC-AUC (Tabla 12). Por eso
**no imputamos**: imputar 39 mil valores de lluvia sería trabajo sin efecto en la predicción. Las 440 filas sin clima sí
se conservan, porque su target y sus variables de calendario se pueden calcular.

## 2. Construcción del target y unión de tablas

**Unidad de análisis:** la pareja (barrio, hora). Se construye la **rejilla completa** 22 barrios × 26.280 horas de
2017-2019. A esa rejilla se le une `clima` y se marca **target = 1** cuando la pareja aparece en `accidentes`. Los
**negativos** son todas las demás parejas de la rejilla. Las parejas posteriores al 2019-07-31 quedan con target = −1:
es el periodo que se debe predecir.""",

"s2_1": r"""**Lectura de la Tabla 6 y la Figura 5:**

1. La rejilla tiene 578.160 parejas: las 577.720 de `clima` más las **440 de las horas sin clima** (1.5).
2. El entrenamiento tiene **497.376 parejas con 9.138 positivas: 1.84 %**. Es el desbalance del reto (sección 6).
3. La evaluación tiene exactamente las **80.256 parejas** de la consulta del README.
4. **Figura 5a:** la prevalencia mensual oscila entre 1.3 % y 2.2 % sin tendencia clara.
5. **Figura 5b:** el **97 %** de las parejas positivas tiene un solo accidente. Un target binario pierde muy poca
   información frente a contar accidentes.

### 2.1 ¿Se corrige el target con los errores de `raw_accidentes`?""",

"s3": r"""**Lectura de la Tabla 7:** corregir el error AM/PM y quitar los 5 accidentes mal geocodificados cambiaría solo 7
parejas positivas (6 desaparecen y 1 aparece), el **0.08 %**.

**Decisión:** se usa la tabla `accidentes` **tal cual**. Según el README, las etiquetas ocultas del periodo de
evaluación son las filas de esa misma tabla, así que el modelo debe aprender exactamente la misma definición de
positivo que se usa para calificarlo. Las correcciones se documentan, pero no se aplican al target.

## 3. Ingeniería de características

Las tablas no traen ninguna variable de calendario (README), así que se construyen desde `TW`. Una regla guía todas
las decisiones: **toda variable debe poder calcularse para diciembre de 2019 con lo que se sabe hoy**. El calendario se
conoce con años de anticipación. Los accidentes de ago-dic 2019 no se conocen, así que no se usan lags del target
dentro del periodo a predecir.""",

"s3_1": r"""**Codificación (Tabla 8):**

1. **Categórica:** `barrio` entra como categórica nativa de LightGBM, que agrupa barrios en cada división sin crear 22
   columnas. En la logística (5.2) se usa *one-hot*.
2. **Cíclicas:** la hora 23 y la hora 0 son vecinas, pero como entero están en los extremos. El seno y el coseno las
   ponen juntas (Figura 6c). Un árbol no lo necesita, porque corta la hora en intervalos (por ejemplo, "h ≥ 6 y h < 19").
   Por eso se calculan y se **comparan** en 5.2, pero no entran en el modelo final.
3. **Festivos:** la lista de festivos de Colombia 2017-2019 (53 días) incluye los trasladados a lunes por la Ley Emiliani
   y la Semana Santa.""",

"s4": r"""**Lectura de la Figura 6 (por qué estas variables):**

1. **(a) El tipo de día cambia la forma del día, no solo su nivel.** Un día laboral tiene su pico a las 17 h (5.35 %
   de las horas-barrio con accidente). Un festivo a esa misma hora tiene 0.69 %, y a las 8 h pasa de 2.59 % a 0.10 %.
   El festivo se parece a un domingo, por eso `tipo_dia` lo codifica como día 6.
2. **(b) Temporada:** enero (1.5 %) y diciembre (1.7 %) son los meses de menor riesgo, por las vacaciones (barras rojas,
   variable `dic_ene`).
3. **(c) Codificación cíclica** de la hora: las 23 h quedan junto a las 0 h.

## 4. Validación

**Partición temporal walk-forward con ventana expansiva** (diapositivas, sección de series de tiempo). Cada corte
entrena con **todo** lo anterior a una fecha y valida en los **5 meses siguientes**: el mismo horizonte de la entrega
(ago-dic 2019). Se usan 6 cortes entre jul-2018 y mar-2019. Uno de ellos (2018-08) evalúa la misma temporada del
leaderboard.""",

"s4_1": r"""**Lectura de la Figura 7 y la Tabla 9:**

1. Cada corte entrena solo con el pasado (barra azul) y se evalúa en las **79.728 a 80.784 parejas** de sus 5 meses
   (barra naranja), con entre 1.391 y 1.533 positivos. La evaluación es siempre el periodo completo, con su desbalance
   real.
2. La última fila es la entrega: entrena hasta el 2019-07-31 y predice ago-dic 2019, que el leaderboard califica.
3. **Expansiva y no deslizante:** en una exploración previa, entrenar solo con los últimos 6, 12 o 18 meses dio 0.7765,
   0.7816 y 0.7824, frente a **0.7842** con todo el historial (5.3). Con 1.84 % de positivos, cada año extra de datos
   pesa más que adaptarse al cambio reciente.

**¿Por qué una partición aleatoria sería un error?** Se compara el mismo modelo con las dos particiones.""",

"s4_2": r"""**Lectura de la Tabla 10:** con partición aleatoria el modelo saca **0.8016**, con temporal **0.7790**. La aleatoria
infla el resultado en **+0.023**. El modelo entrena con horas posteriores a las que evalúa, y con la variable de
tendencia `t` puede aprender el nivel de cada barrio en esas mismas semanas. En el leaderboard nunca tiene esa información: la
partición aleatoria promete un resultado que no se cumple.

## 5. Modelado

### 5.1 ¿Dónde está la señal? Prueba de oráculos

Antes de elegir modelo, una pregunta de fondo: ¿ayudaría saber cuántos accidentes **habrá** ese día, esa semana o ese
mes? Es lo que intentaría un modelo de series de tiempo como **Prophet**, que pronostica el nivel diario con tendencia y
festivos. Para responderla, se hace trampa a propósito: se toma una base simple (tasa barrio × hora × perfil del día de
la semana) y se le da el **número real** de accidentes de ese día, semana o mes en el periodo de validación, **sin contar
la propia fila**. Si ni siquiera con el nivel real se gana, un pronóstico de ese nivel tampoco puede ganar.""",

"s5": "",

"s5_1": r"""**Lectura de la Figura 8 y la Tabla 11:**

1. Conocer el nivel **real** del día o de la semana sube el ROC-AUC apenas **+0.0013** y **+0.0020**. Y eso con
   información perfecta, que nunca se tiene.
2. El nivel real del día en ese barrio, del mes en ese barrio o de esa hora en la comuna **empeora** el resultado
   (−0.0015 a −0.0075). Con tan pocos accidentes, esos niveles son sobre todo ruido.

**Conclusión:** la señal no está en *cuántos* accidentes habrá en un periodo, sino en **dónde y a qué hora**, es
decir en la estructura barrio × hora × tipo de día. Eso explica por qué Prophet, la tendencia por barrio y el clima no
mejoraron el modelo (5.3), y orienta la búsqueda hacia un modelo que capture esa estructura.

### 5.2 La escalera de modelos

Se parte del `groupby` del README y en cada paso se agrega una sola idea, evaluando siempre en los mismos 6 cortes:

| Paso | Idea que agrega |
|---|---|
| M0 | Tasa histórica barrio × hora (la referencia del README) |
| M1 | + tipo de día, con suavizado hacia la tasa barrio × hora (las celdas tienen pocos datos) |
| M2 | Modelo lineal: logística L2 con interacciones barrio × hora, barrio × tipo de día y hora × tipo de día, más el calendario |
| M3 | **LightGBM global**: un solo modelo para los 22 barrios, que aprende las interacciones solo |
| M4 | + tendencia `t` (cambio lento de cada barrio) → **modelo final** |
| M4 + … | Tres ideas que se probaron y se descartaron: clima, lag de 52 semanas, codificación cíclica |""",

"s5_1b": "",

"s5_2": r"""**Lectura de la Tabla 12 y la Figura 9:**

1. **M0 → M1 (0.7606 → 0.7645):** el `groupby` reproduce la referencia del README (0.7589 en el leaderboard). Agregar
   el tipo de día ayuda, pero las celdas barrio × hora × día tienen 2 o 3 accidentes en promedio y el promedio simple
   se queda corto.
2. **M1 → M2 (0.7808):** la logística con interacciones suma el calendario completo y ya queda por encima del valor
   del baseline del profesor (0.7739, medido en el leaderboard).
3. **M2 → M3 (0.7841):** **LightGBM global** gana en los 6 cortes. Es la recomendación de las diapositivas para
   *muchas series relacionadas*: un **modelo global** que aprende lo común (hora pico, festivos, temporada) con los
   9.138 positivos juntos, y usa `barrio` para lo propio de cada barrio. Las interacciones no hay que escribirlas a
   mano: los árboles las encuentran.
4. **M3 → M4 (0.7845):** la tendencia `t` agrega un poco más, y es el **modelo final**. Rango entre cortes:
   0.7790–0.7894.
5. **Descartes:**
   - **El clima empeora** (0.7827): confirma 1.6.
   - **El lag de 52 semanas** (0.7847) y **la codificación cíclica** (0.7850) quedan +0.0002 y +0.0005 por encima de M4.
     Es una diferencia muy chica frente a lo que varía M4 entre cortes (0.010) y frente al rango por azar de un periodo
     de 5 meses (±0.012, sección 8), así que es ruido. Se mantuvo M4, el modelo elegido y entregado antes de esta
     comparación.

**¿Por qué LightGBM?** (1) Hay 22 series con pocos positivos cada una, y un modelo global comparte información entre
ellas. (2) Captura interacciones no lineales (un festivo apaga la hora pico, Figura 6a) sin especificarlas. (3) Maneja
`barrio` como categórica nativa. (4) Entrena en segundos sobre medio millón de filas.

### 5.3 Lo que probamos y no funcionó

Estas ideas se midieron en exploraciones previas con la misma validación walk-forward, usando scripts del repositorio.
No se re-ejecutan aquí para no alargar el notebook. Cada fila compara con la referencia **de ese mismo experimento**.

| Idea | Qué se probó | ROC-AUC (referencia) | Por qué no funcionó |
|---|---|---|---|
| **Prophet** por barrio | Pronóstico del nivel diario de cada barrio (festivos de Colombia, estacionalidad semanal y anual) como variable de LightGBM | 0.7614 (0.7811) | El nivel diario no aporta (5.1). La serie diaria por barrio es 68 % ceros, y Prophet asume ruido gaussiano |
| Ventana deslizante | Entrenar solo con los últimos 6 / 12 / 18 meses | 0.7765 / 0.7816 / 0.7824 (0.7842) | Con 1.8 % de positivos, perder historia cuesta más que adaptarse al cambio |
| Lags estacionales | Target de 52 y 104 semanas antes, puntual y suavizado | 0.7841–0.7846 (0.7842) | Un barrio-hora concreto tiene ~2 % de probabilidad: su valor de hace un año es casi un volado |
| *Target encoding* | barrio × mes y barrio × tipo de día, fuera de muestra por trimestre | 0.7838–0.7844 (0.7842) | LightGBM ya aprende esas interacciones |
| Variables espaciales | Centroide, % en intersección o glorieta, % con heridos, tasa de los 3 barrios vecinos | 0.7835 (0.7842) | `barrio` ya identifica todo lo estático |
| Clima diario | Lluvia del día, lluvia de las 1/3/6 h previas | 0.7828–0.7839 (0.7842) | Igual en los 22 barrios y sin señal propia (1.6) |
| Ensamble | Logística + LightGBM, promedio de rangos | 0.7841 (0.7842) | Los dos modelos ven la misma estructura: se equivocan en lo mismo |
| Optuna | 40 intentos (TPE), con un corte de prueba limpio que la búsqueda no veía | 0.7786 (0.7791) | El modelo es poco sensible a los hiperparámetros; la ganancia en ajuste era ruido |
| Objetivo Poisson | Predecir el número de accidentes | 0.7840 (0.7842) | El 97 % de los positivos tiene un solo accidente (Figura 5b) |
| *Bagging* | 20 LightGBM con *bootstrap* por semanas | 0.7841 (0.7842) | LightGBM regularizado ya tiene poca varianza |
| Pendiente por barrio | Tendencia propia de cada barrio (Poisson mensual) + Optuna | 0.7842–0.7846 (0.7851) | Unos 20 puntos mensuales por barrio, ruidosos; extrapolar 5 meses empeora |
| **Un modelo por barrio** | 22 LightGBM locales (complejidad fija y ajustada), estandarizados al nivel del global, y en ensamble con el global | 0.7690–0.7746 (0.7851) | Cada barrio tiene ~400 positivos: el modelo local aprende ruido. El ensamble con el global solo empeora a medida que pesa más el local |""",

"s6": r"""## 6. Manejo del desbalance

Con 1.84 % de positivos se compararon las técnicas de balanceo de clase sobre el modelo final, en los cortes 2018-08 y
2019-03. Se evalúan tres cosas: ROC-AUC, PR-AUC y **la probabilidad media que predice el modelo**, porque una
probabilidad bien calibrada es la que permite fijar un umbral con costos reales (6.1).""",

"s6_1": r"""**Lectura de la Figura 10 y la Tabla 13:**

1. **Ninguna técnica mejora el orden.** Sin rebalanceo: ROC-AUC **0.7813** y PR-AUC **0.0659**. *Undersampling* empata
   (0.7813 / 0.0658), `class_weight` y el *bootstrap* de positivos quedan levemente abajo, y **SMOTENC es el peor**
   (0.7785 / 0.0646). Es el mismo patrón de la tabla del profesor en las diapositivas: el remuestreo apenas mueve el
   PR-AUC.
2. **Todas destruyen la calibración (Figura 10c).** Sin rebalanceo, el modelo predice en promedio **1.78 %** frente a
   una prevalencia real de 1.85 %. Con `class_weight` predice 35.9 %, con *bootstrap* 22.8 % y con *undersampling*
   9.0 %.
3. **Por qué:** el ROC-AUC solo mide el *orden*, y replicar o eliminar filas no crea información nueva sobre qué
   barrio-hora es más riesgoso. SMOTENC crea positivos sintéticos con el día del año y la tendencia interpolados
   entre vecinos: combinaciones que nunca existieron y que empeoran el orden.

**Decisión:** entrenar **sin rebalanceo** y manejar el desbalance donde importa: en el **umbral de decisión**.

### 6.1 Umbral de decisión por costo

Se asigna un costo a cada error: **C_FP = 1**, una hora de patrulla en un barrio donde no pasa nada, y **C_FN = k**, un
accidente que ocurre sin agente cerca. El umbral óptimo minimiza FP·1 + FN·k. Para que el umbral no quede ajustado a
los mismos datos, se **elige en el corte 2018-08** y se **aplica sin cambios al corte 2019-03**.""",

"s6_2": r"""**Lectura de la Figura 11 y la Tabla 14:**

1. **El umbral óptimo coincide con la teoría.** Si las probabilidades están calibradas, el óptimo es 1/(1+k). Con k = 25
   el umbral elegido es **0.034** y la fórmula da 0.038. Es la prueba práctica de que no rebalancear dejó probabilidades
   útiles.
2. **El umbral se sostiene fuera de muestra.** Con k = 25, en 2019-03 se mantiene el recall (**0.546**) y el ahorro
   frente a no patrullar pasa de 21.7 % a **17.8 %**.
3. **Con k = 10 no conviene patrullar:** el ahorro es −0.3 % en 2019-03. Un agente tiene que evitar algo que cueste
   bastante más que su hora para que el modelo compense.

**Escenario elegido: C_FN / C_FP = 25.** El 35 % de los accidentes de `raw_accidentes` deja heridos y 29 dejan muertos,
así que un accidente evitable cuesta mucho más que una hora de patrulla. 25 es un supuesto conservador. La Figura 11
muestra cómo cambia el umbral si la Secretaría valora distinto ese costo: con k = 50 o 100 conviene alertar más
(27 % a 50 % de las horas-barrio).""",

"s7": r"""## 7. Métricas más allá del ROC-AUC

Modelo final, con el umbral de 6.1 (0.0342, k = 25), en los dos cortes de validación.""",

"s7_1": r"""**Lectura de la Figura 12 y la Tabla 15:**

1. **Matriz de confusión (Figura 12c-d):** en 2019-03, de 1.448 accidentes el modelo anticipa **790** (recall 0.546) y
   genera 13.300 falsas alarmas. Precision **0.056**: aproximadamente 1 de cada 18 alertas tiene un accidente.
2. **Recall 0.546 con 17 % de las horas alertadas:** vigilando 1 de cada 6 horas-barrio se cubre más de la mitad de los
   accidentes.
3. **La accuracy engaña:** el modelo tiene 0.83 y la regla "nunca hay accidente" tiene **0.98**, sin servir para nada.
   Por eso el README descarta la accuracy.
4. **ROC frente a PR (Figura 12a-b):** la curva ROC se ve "buena" (0.78) porque premia ordenar bien a los 98 %
   negativos. La curva PR mide lo que le interesa a quien envía patrullas, y muestra que la precisión arranca en ~10 % y
   baja de forma gradual. El PR-AUC de 0.060–0.072 parece bajo, pero hay que compararlo con el azar, que es igual a la
   prevalencia (0.019): el modelo es **3.3 a 3.8 veces mejor que el azar**. El ROC-AUC dice que el modelo *ordena*
   bien; la PR dice que, aun ordenando bien, **la mayoría de alertas no tendrán accidente**, porque el evento es raro.
   El modelo sirve para **priorizar**, no para anticipar el accidente concreto.""",

"s8": r"""## 8. Resultado oficial

El modelo final (M4, `random_state = 0`) se reentrena con **todo** el entrenamiento (hasta 2019-07-31) y predice las
80.256 parejas de la consulta del README. El archivo resultante se compara con el que se subió al leaderboard.""",

"s8_1": r"""**Lectura:**

1. El archivo tiene exactamente las **80.256 filas** de la consulta del README, ids únicos y probabilidades entre 0 y 1.
2. Coincide con el archivo que se subió al leaderboard (`mi_prediccion_Savelezc.csv`) con una **diferencia máxima de
   1e-16**: es el mismo archivo. Este notebook reproduce exactamente el resultado oficial.
3. **Tabla 16:** la probabilidad media entregada es **1.90 %**, cerca de la prevalencia histórica (1.84 %), con un
   rango de 0.07 % a 16.1 %.

¿Qué tan bien predijo la validación el resultado del leaderboard? Además del promedio de los 6 cortes, se estima cuánto
puede variar el ROC-AUC **por puro azar** en un periodo de 5 meses: se remuestrean 1.000 veces los días del corte
2018-08, que es la misma temporada.""",

"s9": r"""**Lectura de la Figura 13 y la Tabla 17 — RESULTADO OFICIAL: 0.7838.**

1. **La validación anticipó el resultado.** El leaderboard dio **0.7838**: 0.0007 por debajo del promedio de los 6
   cortes (0.7845) y prácticamente igual al corte de la misma temporada (0.7836). Cae en el centro del rango del 95 % por
   azar, [0.7715, 0.7950]. La validación walk-forward fue una estimación honesta.
2. **Frente a las referencias del README:** +0.0249 sobre la tasa histórica (0.7589) y **+0.0099 sobre el baseline del
   profesor (0.7739)**. Eso ubica la etapa 1 en el tramo de 90–100 %.
3. **Figura 13b, qué usa el modelo:** `barrio` (41 %) y `h` (40 %) explican más del 80 % de las divisiones, seguidos de
   `tipo_dia` (10 %). Es la estructura barrio × hora × tipo de día que señaló la prueba de oráculos (5.1). La tendencia
   `t`, el día del año y los festivos refinan; las banderas de temporada casi no se usan.

## 9. Caso de uso y limitaciones

### 9.1 Mapa de calor para la operación""",

"s9_0": r"""**Lectura de la Figura 14:** la Secretaría puede ver de un vistazo **dónde y cuándo** concentrar agentes.

1. **Día laboral:** el riesgo se concentra entre las **7 h y las 18 h** en Villa Carlota, Manila, La Aguacatala y El
   Poblado, y el pico más alto es a las 16–17 h. En la madrugada el riesgo es mínimo en todos los barrios.
2. **Fin de semana o festivo:** el riesgo baja a la mitad o menos y se corre al mediodía (11–15 h).
3. Los barrios con menos accidentes históricos (Lalinde, Altos del Poblado, El Diamante 2) tienen riesgo bajo a
   cualquier hora.

### 9.2 Turnos: N patrullas de 4 horas

Siguiendo el caso de brigadas de las diapositivas: cada día se cubren las **4·N parejas barrio-hora de mayor
probabilidad**. Se mide en el corte de validación 2019-03, el más reciente con etiquetas.""",

"s9_1": r"""**Lectura de las Tablas 18 y 19:**

1. **Una patrulla de 4 horas** cubre el 0.8 % de las horas-barrio y el **3.8 %** de los accidentes: **5 veces** más que
   patrullar al azar. Con **10 patrullas** (7.6 % de las horas) se cubre el **28.7 %**.
2. El *lift* baja al agregar patrullas (de 5.0 a 3.8): las primeras toman las horas más riesgosas, y las siguientes
   horas ya son menos riesgosas.
3. **Tabla 19, ejemplo de un turno real:** para el lunes 5 de agosto de 2019, las horas prioritarias son la tarde en
   **Villa Carlota** (12–18 h, hasta 14.4 % de probabilidad), **Manila** y **La Aguacatala** (16–17 h). Un plan concreto
   sería una patrulla en Villa Carlota de 14 a 18 h y otra rotando entre Manila y La Aguacatala de 15 a 18 h.

### 9.3 Limitaciones y sesgos

1. **Solo accidentes reportados.** Los choques simples que se arreglan sin reporte no están. El modelo aprende dónde
   *se reporta*, que puede no ser exactamente dónde *ocurre*.
2. **Riesgo de retroalimentación.** Si se patrulla donde el modelo dice, allí se registrarán más incidentes, y el
   siguiente modelo reforzará esos mismos barrios. Conviene reservar parte de las patrullas para rotación aleatoria.
3. **Precisión baja por naturaleza:** ~6 % en el umbral elegido. El accidente concreto no es predecible con estos datos;
   el valor del modelo es **priorizar** y no anticipar cada evento.
4. **Información que no está en los datos:** eventos masivos, obras, cierres viales y volumen de tráfico en tiempo real.
   Serían las variables con más potencial para ir más allá de ~0.78.
5. **Calidad de la fuente:** errores de geocodificación (5 casos) y horas mal registradas (1 caso). Son pocos, pero
   muestran que la ubicación y la hora no son perfectas.
6. **Cambio en el tiempo:** algunos barrios cambian de nivel año a año. El modelo debe **reentrenarse cada mes** con los
   datos nuevos, porque 5 meses es el horizonte más largo que se validó.""",

"resumen": r"""## Resumen

| Punto del README | Decisión | Evidencia |
|---|---|---|
| Exploración y calidad | 20 horas sin clima, bloque sin lluvia en 2018, `HORA` en dos formatos, 1 error AM/PM, 5 accidentes mal geocodificados | Tablas 1-4, Figuras 1-3 |
| ¿Imputar clima? | **No**: el clima no separa accidentes dentro de barrio × hora y empeora el modelo | Figura 4, Tabla 5, Tabla 12 |
| Target y unión | Rejilla completa barrio × hora; target desde la tabla oficial `accidentes`; negativos = resto de la rejilla | Tablas 6-7, Figura 5 |
| Variables | Barrio categórico + calendario (festivos de Colombia, tipo de día, temporadas) + tendencia; cíclicas comparadas | Tabla 8, Figura 6 |
| Validación | Walk-forward, ventana expansiva, 6 cortes de 5 meses; la aleatoria infla +0.023 | Figura 7, Tablas 9-10 |
| Modelo | **LightGBM global**: la señal está en barrio × hora × tipo de día, no en el nivel del periodo | Figuras 8-9, Tablas 11-12 |
| Desbalance | Sin rebalanceo (ninguna técnica mejora y todas descalibran); umbral por costo (k = 25 → 0.034) | Figuras 10-11, Tablas 13-14 |
| Métricas | Recall 0.55, precision 0.06, PR-AUC 3.3-3.8 veces el azar | Figura 12, Tabla 15 |
| **Resultado oficial** | **ROC-AUC = 0.7838** (validación 0.7845), por encima del baseline del profesor (0.7739) | Figura 13, Tablas 16-17 |
| Caso de uso | Mapa de calor por barrio y hora; 1 patrulla de 4 h cubre 5 veces más accidentes que al azar | Figura 14, Tablas 18-19 |""",
}
