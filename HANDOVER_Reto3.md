# HANDOVER — Reto 3: Sistema de recomendación de películas

*Lo escribe la sesión que terminó el Reto 2 (2026-10-17). Léelo completo antes de hacer cualquier cosa.*

---

## 0. Lo primero que debes hacer (en este orden)

1. **Leer el README REAL del reto, no este resumen.** El repo del curso es público:
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/santiagovelcasallas/ml-course /home/user/santiagovelcasallas/ml-course
   ```
   - Leer `assignments/recomendador-peliculas/README.md` y `starter_notebook.ipynb`.
   - Leer `assignments/README.md`: tiene reglas generales. El notebook debe estar **documentado**, con
     **análisis de errores e interpretación**.
   - Leer `notebooks/sesion_04_sistemas_recomendacion/clase.ipynb`, el notebook de clase de recomendación. Trae una
     factorización con SGD en numpy y avisa contra los splits aleatorios.
   - Si `add_repo` está disponible, úsalo con `access: "read"`. En el Reto 2, la primera vez que lo intenté **sin que
     el usuario lo pidiera**, el clasificador lo bloqueó. Si pasa, explícaselo al usuario.
2. **Verificar que los datos bajan.** `https://d3qixogk4zgixq.cloudfront.net/data/recomendador-peliculas/train.csv`.
   En el Reto 2 la red del entorno bloqueaba `d3qixogk4zgixq.cloudfront.net` y el usuario tuvo que agregarlo en
   *Editar entorno → Acceso a la red → dominios permitidos*. Si da 403 en el proxy, díselo en **dos líneas**: qué host
   y dónde se cambia.
3. **Repo de trabajo:** `santiagovelcasallas/Data_Science_example_01`, rama `claude/new-session-hjbq1o` (o la que
   indique la sesión). Al principio el push dio 403 porque la app de Claude no estaba autorizada. El usuario la
   autorizó y ahora funciona. Trabajar en una carpeta `reto3/`.
4. **Fecha límite del leaderboard: 20 de octubre de 2026, 11:59 PM, hora Colombia.** Revisa la hora con
   `TZ=America/Bogota date` y avísale al usuario cuánto falta si el tiempo aprieta.

---

## 1. El Reto 3 en una página (del README real, leído el 2026-10-17)

- **Tarea:** predecir la calificación (1 a 5) que un usuario le da a una película. Datos MovieLens 100k: 943 usuarios
  y 1.682 películas.
- **Archivos:** `train.csv` (79.619 filas: `id, user_id, item_id, rating, timestamp`), `test.csv` (20.381 pares sin
  rating) y `movies.csv` (título, fecha de estreno, 19 géneros). Se leen directo de
  `https://d3qixogk4zgixq.cloudfront.net/data/recomendador-peliculas/{train,test,movies}.csv`.
- **Corte temporal POR USUARIO:** el test es el **20 % más reciente de cada usuario**. "Para predecir lo que un usuario
  calificó en marzo, solo pueden usar lo que ese usuario calificó **antes**". Una partición aleatoria infla el resultado.
- **Cold-start:** 69 películas del test no aparecen en train (86 filas, 0.4 %). Solo hay `movies.csv` (género y año).
  "Decidir qué hacer con ellos y justificarlo es parte del trabajo" (vale 10 % de la nota).
- **Entrega:** CSV `id,rating`, 20.381 filas. El rating es **continuo, sin redondear**, y debe estar entre 1 y 5. El
  starter trae `construir_submission(ids, predicciones)`, que valida filas, duplicados, NaN y rango: **úsala** (el
  usuario lo exigió en el Reto 2).
- **Métrica: RMSE** (más bajo es mejor). Referencias: media global **1.2079**; media + sesgo de usuario + sesgo de
  película **1.0139**; **baseline del profesor (factorización de matrices) 0.9683**.
- **Nota:** 40 % leaderboard (≤ 0.9683 → 90-100 %; top-3 → 100 %) + 60 % notebook. Rúbrica del notebook:

  | Sección | Peso | Qué se evalúa |
  |---|---|---|
  | Exploración | 10 % | Distribución de calificaciones, dispersión de la matriz, usuarios y películas más activos, sesgos observados |
  | Baseline de sesgos | 10 % | Implementado y medido **antes** de cualquier modelo complejo |
  | Modelos de recomendación | 15 % | **Al menos dos enfoques** (vecindario, factorización, híbrido), comparados con criterio |
  | Validación | 10 % | Cómo se montó el corte temporal por usuario dentro de train; por qué no sirve una aleatoria |
  | **Cold-start** | 10 % | Qué se hizo con las películas sin historial y por qué |
  | Interpretación y caso de uso | 5 % | Qué recomendaría el sistema, limitaciones, qué falta para producción |

  Además, `assignments/README.md` pide un notebook **documentado** con **análisis de errores e interpretación**.
- **Envío** (lo hace el usuario, con su `.env`): `uv run scripts/submit.py recomendador-peliculas mi_prediccion.csv` y
  `uv run scripts/check_status.py recomendador-peliculas`. Cuenta la última entrega. La API key es del usuario: nunca
  la pidas ni la manejes.

---

## 2. El usuario: cómo trabajar con él (lo aprendido a la mala en el Reto 2)

**Quién es:** Santiago Vélez Casallas, Maestría en Ciencia de Datos, EAFIT. Curso Aprendizaje Automático 2026-2,
profesor Pablo Saldarriaga. Todo se hace **en español**.

### Reglas de comunicación (no negociables)
1. **Explica siempre filas y columnas de cada tabla.** Si una tabla dice "A, B, R1…" o "promedio", define qué es cada
   cosa *antes* de la tabla. Ejemplo de lo que molestó: mostrar "fold A, B, R1" sin explicar.
2. **Nunca un número sin su naturaleza.** Separa siempre **validación** (estimación) de **leaderboard** (oficial). Da
   desde el principio el **rango de incertidumbre** (bootstrap), no un número puntual que parezca promesa. En el
   Reto 2 se repitió "0.785" muchas veces sin decir que era una estimación con ±0.012, y el usuario lo vivió como
   "datos inflados". El leaderboard dio 0.7838, dentro del rango, pero la confianza ya se había dañado.
3. **No inventes ni adivines.** Si no leíste algo (el README, una cifra), dilo. "Lo leí del handover" no es leer el
   README.
4. **Respuestas cortas** cuando las pide ("respuesta corta"). Sin chachara ni disculpas largas. Ve al punto.
5. **No generes ni sugieras entregas** que el usuario no pidió. En el Reto 2 dijo varias veces "no generes entrega,
   estamos experimentando". Cuando la pida, genérala; si el plazo aprieta, avísale una vez.
6. **No discutas que una meta sea imposible.** Muéstrale la evidencia (experimentos) y que él decida. Sí puedes decir
   con datos dónde está el techo.
7. **No cambies nada a escondidas.** Ejemplo real: se fijó `random_state=0` en la entrega sin avisar y cambió el AUC de
   validación de 0.7845 a 0.7836. Eso generó desconfianza. Cualquier cambio de semilla, parámetro o archivo se anuncia.
8. Cuando un experimento corre en segundo plano, avisa en una línea y sigue. No llenes el chat de "1 de 6 folds listo".
9. **Commits:** **NUNCA** agregar `Co-Authored-By: Claude` ni líneas de atribución en los commits del usuario. Mensajes
   de commit en español.

### Forma de experimentar que le gustó
- **Primero scripts de experimento** (`reto3/exp_*.py`), resultados en tablas con definición de filas y columnas, y
  **después** el notebook.
- Cada experimento: qué se prueba, tabla de resultados, lectura corta, conclusión ("sirve" / "no sirve, por qué").
- Diagnosticar el *por qué* cuando algo falla. Le gustó mucho la prueba de **oráculos** del Reto 2: "aun con el nivel
  REAL del día no se gana, entonces ningún pronóstico de ese nivel puede ganar".
- Le gusta pensar en voz alta ideas como "un modelo por barrio" o "bootstrapear". Pruébalas con honestidad, en la
  misma validación, y explica con datos por qué funcionan o no.

---

## 3. El notebook: estilo y requisitos (Reto 1 y Reto 2 como plantilla)

**Plantilla de referencia:** `reto2/SantiagoVelez_Reto2_Accidentalidad_Poblado.ipynb` en el repo de trabajo. Su
generador es `reto2/build_notebook.py` (celdas de código) junto con `reto2/notebook_md.py` (markdown). El usuario
también compartió su notebook del Reto 1 (tasación de diamantes), que sirvió de plantilla de estilo.

### Estructura
1. **Portada:** título, curso, profesor, autor, objetivo, métrica, tabla de referencias del README y el recuadro
   `> ### RESULTADO OFICIAL: RMSE = **x.xxxx**` (se llena cuando el usuario diga el score del leaderboard).
2. **Celda de entorno:** instala solo lo que falte, con pip o con `uv pip` (el entorno uv del curso no trae pip).
   Funciona en Colab, en uv y en Python normal. El `pyproject.toml` del curso trae numpy, pandas, scikit-learn,
   matplotlib, jupyter, requests, python-dotenv y optuna. **No trae seaborn** (y en el Reto 2 tampoco lightgbm ni
   imbalanced-learn).
3. **Secciones numeradas en el orden de la rúbrica del README**, punto por punto.
4. **Cada celda de código produce una figura o una tabla numerada** ("Figura N — título", "Tabla N — título", con la
   función auxiliar `tabla(df, n, titulo)`). Cero celdas sin sentido.
5. **Cada figura o tabla va seguida de un markdown "Lectura de la Figura N / Tabla N"**: lista numerada, autocontenida,
   con las cifras reales, que cite la figura o tabla. **El markdown se escribe DESPUÉS de ver la salida real**, nunca
   "si sale esto, significa aquello". Procedimiento: (1) escribir el código, (2) ejecutar el notebook, (3) leer las
   salidas e imágenes, (4) redactar el markdown con esos números, (5) volver a ejecutar y verificar.
6. **Historia:** por qué se eligió cada cosa, la "escalera" de modelos (del baseline al final, una idea por paso) y una
   tabla de **"lo que probamos y no funcionó"**, que puede venir de scripts y no re-ejecutarse (así se hizo en los
   Retos 1 y 2, diciéndolo explícitamente).
7. **Código comentado** en todas las celdas. El usuario lo reclamó; el `assignments/README.md` pide "documentado".
8. **Análisis de errores** (sección propia) e **interpretación**.
9. **Resultado oficial:** validación frente a leaderboard, con rango por bootstrap y referencias del README.
10. **Caso de uso y limitaciones.**
11. **Resumen final:** tabla "Punto del README | Decisión | Evidencia (Figura/Tabla)".
12. La entrega se genera **dentro del notebook** con `construir_submission` del starter. Si el usuario sube el CSV, se
    verifica que el notebook lo reproduce exactamente (md5 o diferencia máxima).
13. Semilla fija (`SEED = 0`) desde el inicio. El notebook completo debe correr en pocos minutos en Colab.

### Gráficas
- seaborn y matplotlib. Paleta fija usada en el Reto 2 (validada con la skill `dataviz`): AZUL `#2a78d6`, NARANJA
  `#eb6834`, VERDE `#1baf7a`, AMARILLO `#eda100`, VIOLETA `#6250d6`, ROJO `#e34948`, GRIS `#8a8984`.
- Antes de redactar, **mirar cada imagen**: leyendas encima de datos, ejes amontonados, escalas log ilegibles o colores
  saturados son errores que hubo que corregir en el Reto 2.
- Diapositivas del curso: sesión 4 (sistemas de recomendación). El PDF del Reto 2 excluía a propósito las páginas
  53-88, que eran justamente las de Recomendación. Pídele al usuario esas diapositivas si las quieres usar.

### Cómo editar el notebook sin dañar lo que ya funciona (lo pidió explícitamente)
- Herramientas en `herramientas/`:
  - `check_ast.py notebook_antes.ipynb notebook_despues.ipynb` compara el **AST** de cada celda de código. Si el AST es
    igual, solo cambiaron comentarios. Lista las celdas con código modificado y las nuevas.
  - `comentar_reto2.py` es el ejemplo de cómo se insertaron 105 comentarios con un diccionario {línea exacta →
    comentario}.
- Protocolo: (1) copia del notebook anterior, (2) cambios, (3) `check_ast.py`, (4) re-ejecutar con
  `jupyter nbconvert --to notebook --execute --inplace`, (5) comparar las salidas de texto antes y después, (6) md5 del
  CSV de entrega, (7) cero errores y cero markdown vacíos, (8) numeración de figuras y tablas continua y todas citadas.

---

## 4. Validación honesta: lo que más importa (y lo que salió mal antes)

- **Reproducir dentro de train el mismo protocolo del test:** de cada usuario, su 20 % más reciente de train como
  validación, y entrenar con el resto. Para un rango más robusto se puede hacer *walk-forward por usuario*: varios
  cortes por percentil temporal de cada usuario (por ejemplo, validar el tramo 60-80 % entrenando con lo anterior, y el
  80-100 % de train entrenando con todo lo anterior).
- **Demostrar con números** que la partición aleatoria es optimista: mismo modelo, aleatoria frente a temporal. En el
  Reto 2 esa demostración (+0.023) gustó mucho.
- **Evaluar siempre sobre el conjunto completo** de validación. El usuario preguntó explícitamente por esto: en un
  examen anterior "salía una métrica mejor a la real". Verifícalo con un script (en el Reto 2 fue `check_evaluacion.py`)
  y muéstralo en una tabla.
- **Bootstrap por usuario** del RMSE de validación, para dar el rango esperado del leaderboard desde el principio.
- Cuando el usuario pregunte "¿cuánto da en el test?", la respuesta honesta es: **solo el leaderboard lo sabe**. Las
  etiquetas están ocultas y lo nuestro es una estimación.
- **La regla de información disponible:** para cada predicción, solo historial anterior. Ojo con *leaks* sutiles,
  documentados tabla por tabla, como se hizo en el Reto 2, sección 3.1:
  - **Usar los pares de `test.csv` como feedback implícito** (qué películas vio cada usuario en su periodo de test,
    estilo SVD++ del Netflix Prize). Es información del futuro del usuario: **no usarlo**, o si se prueba, consultarlo
    con el usuario y documentarlo como dudoso.
  - Estadísticas de una película calculadas con calificaciones posteriores, del mismo usuario o de otros. Debatible:
    el corte es por usuario, así que otros usuarios pueden tener calificaciones posteriores en train. Documentar la
    decisión.
  - El año de estreno de `movies.csv` es legítimo.

---

## 5. Ideas técnicas para el Reto 3 (sugerencias, no hechos medidos)

1. **Baseline de sesgos regularizado** (media + b_u + b_i con λ, ajustado por ALS o por promedios con suavizado).
   Debe acercarse al 1.0139 del README. Ajustar λ en la validación temporal.
2. **Factorización (SVD de Funk con SGD, en numpy)**: el notebook de clase la trae. Ajustar factores, λ y épocas.
   Meta: ≤ 0.9683. Probar también SVD con sesgos y **ALS**.
3. **Vecindario** (item-kNN con similitud ajustada por baseline, shrinkage). Cuenta como segundo enfoque de la rúbrica.
4. **Híbrido / ensamble lineal** de los anteriores, con pesos ajustados en la validación temporal.
5. **Efectos temporales** (permitidos): deriva del sesgo del usuario en el tiempo, antigüedad de la película al
   momento de calificar (timestamp menos fecha de estreno) y popularidad acumulada hasta ese momento.
6. **Cold-start:** fallback de contenido con media global + sesgo del usuario + sesgo por género y década (ridge sobre
   los 19 géneros y el año), validado simulando películas frías. Son 86 filas: el efecto en el RMSE es pequeño, pero se
   evalúa con 10 % de la nota.
7. **Recortar** las predicciones a [1, 5] y **no redondear**.
8. Librerías: preferir **numpy/scipy/sklearn**. `scikit-surprise` suele fallar al compilar en Python 3.12 o superior;
   si se usa, la celda de instalación debe manejarlo y debe haber un plan B en numpy. Python del contenedor: 3.13.
9. **Análisis de errores:** RMSE por actividad del usuario (pocas o muchas calificaciones), por popularidad de la
   película, por calificación real (los extremos 1 y 5 suelen fallar más), por género y para cold-start.

---

## 6. Resultado y aprendizajes del Reto 2 (contexto)

- **Leaderboard: ROC-AUC 0.7838** (baseline del profesor 0.7739). Validación walk-forward: 0.7845 en promedio de 6
  cortes. Modelo final: LightGBM global con barrio, calendario y tendencia, `random_state=0`.
- El archivo subido (`mi_prediccion_Savelezc.csv`, md5 `c3ecf584ede8117d5ca91df8c721e49a`) lo reproduce exactamente el
  notebook. **Ojo:** `reto2/mi_prediccion.csv` del repo es otra versión, con la semilla por defecto. No es la entregada.
- Lo que no funcionó (para no repetir el patrón): Prophet, lags, ventana deslizante, target encoding, Optuna, modelos
  por barrio, SMOTE y bootstrap de positivos. El techo estaba en la información disponible y no en el modelo.
  **Lección:** medir pronto *dónde está la señal* (oráculos, rango por bootstrap) antes de quemar horas en variantes.
- Hubo 17 familias de experimentos y quemaron muchos tokens del usuario. Sé eficiente: menos experimentos, mejor
  elegidos, y dile el costo antes de lanzar algo largo.

---

## 7. Checklist final antes de entregar el notebook del Reto 3

- [ ] Cada punto de la rúbrica del README tiene su sección, en orden, con figura o tabla numerada y su lectura.
- [ ] Baseline de sesgos construido y medido antes que todo lo demás.
- [ ] Al menos dos enfoques comparados en la misma validación temporal por usuario.
- [ ] Demostración numérica de por qué la partición aleatoria es un error.
- [ ] Cold-start: qué se hizo, por qué y cómo se validó.
- [ ] Análisis de errores e interpretación (assignments/README).
- [ ] La regla de información disponible, documentada tabla por tabla.
- [ ] Todas las celdas de código comentadas.
- [ ] `construir_submission` del starter; 20.381 filas; ratings continuos en [1, 5].
- [ ] Resultado oficial del leaderboard frente a validación y rango por bootstrap; referencias del README.
- [ ] Caso de uso (qué recomendaría a un usuario concreto), limitaciones y qué falta para producción.
- [ ] Resumen final; el notebook corre de principio a fin sin errores y en Colab; se verifica que reproduce el CSV
      subido.
