"""Inserta comentarios encima de lineas exactas de build_notebook.py. Solo agrega lineas que empiezan con '#'."""
import sys
C = {
# instalación
'REQUERIDAS = {"lightgbm": "lightgbm", "seaborn": "seaborn", "imblearn": "imbalanced-learn"}':
    "Librerías que no vienen en el pyproject del curso: módulo a importar -> paquete a instalar",
'faltan = [pkg for mod, pkg in REQUERIDAS.items() if importlib.util.find_spec(mod) is None]': "Solo se instala lo que falta",
'if importlib.util.find_spec("pip") is not None:': "Colab o Python normal (con pip)",
'elif shutil.which("uv"):': "Entorno uv del curso (no trae pip)",
# imports
'import os, sqlite3, urllib.request, warnings': "Librerías estándar, análisis de datos y gráficas",
'from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, confusion_matrix':
    "Métricas: ROC-AUC (la del leaderboard), PR-AUC, curvas y matriz de confusión",
'SEED = 0': "Semilla fija: todo el notebook es reproducible",
'AZUL, NARANJA, VERDE, AMARILLO, VIOLETA, ROJO, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948", "#8a8984"':
    "Paleta fija para que cada color signifique lo mismo en todas las figuras",
'def tabla(df, n, titulo):': "Muestra una tabla con su número y título, para citarla desde el texto",
# carga
'URL = ("https://d3qixogk4zgixq.cloudfront.net/data/prediccion-accidentalidad-poblado/"': "Base de datos del reto (~85 MB); se descarga una sola vez",
'con = sqlite3.connect(DB)': "Se leen las tres tablas completas, tal como vienen de la fuente",
# 1.1
'resumen = pd.DataFrame({': "Resumen de cada tabla: tamaño, rango de fechas, barrios y faltantes",
'print("Mismos 22 barrios, escritos igual, en las tres tablas:", set(clima.BARRIO) == set(acc.BARRIO) == set(raw.BARRIO))':
    "Consistencia entre tablas: nombres de barrios y parejas (barrio, hora) repetidas",
# 1.2
'for d in (clima, acc, raw):': "TW llega como texto: se convierte a fecha-hora en las tres tablas",
'm = raw.groupby(raw.TW.dt.to_period("M")).size()': "(a) Número de accidentes por mes",
'hm = acc.groupby([acc.TW.dt.dayofweek, acc.TW.dt.hour]).size().unstack(fill_value=0)': "(b) Horas-barrio con accidente por día de la semana (filas) y hora (columnas)",
'por_ano = raw.groupby(raw.TW.dt.year).size()': "Cifras citadas en el texto",
# 1.3
'fuera = raw.Lon < -75.62': "Accidentes con longitud fuera de El Poblado (la comuna está entre -75.59 y -75.54)",
'cnt = raw.BARRIO.value_counts().sort_values()': "(a) Accidentes por barrio",
'cent = raw[~fuera].groupby("BARRIO")[["Lon", "Lat"]].median()': "(b) Centro de cada barrio (mediana de sus accidentes) para poner la etiqueta",
'ax[2].scatter(raw.Lon[~fuera], raw.Lat[~fuera], s=3, alpha=.25, color=AZUL, label="El Poblado")': "(c) Vista completa, con los puntos mal geocodificados en rojo",
# 1.4
'falt = (clima.groupby(clima.TW.dt.to_period("M"))[VARS_CLIMA].apply(lambda g: g.isna().mean() * 100)).T':
    "(a) % de filas faltantes de cada variable en cada mes",
'sns.boxplot(x=clima.TW.dt.hour, y=clima.temperature, color=AZUL, fliersize=0, linewidth=.8, ax=ax[1])': "(b) Ciclo diario de la temperatura",
'nan = clima[VARS_CLIMA].isna().sum()': "Tabla 2: faltantes por variable",
'calma = (clima.windBearing.isna() & (clima.windSpeed == 0)).sum()': "windBearing (dirección) faltante cuando no hay viento",
'bloque = clima[clima.precipIntensity.isna()].TW': "¿El faltante de lluvia es por barrio o de toda la comuna a la vez?",
'print("Correlación de la temperatura entre barrios a la misma hora:",': "¿El clima cambia entre barrios? Correlación media entre los 22 a la misma hora",
# 1.5
'h = raw.HORA.astype(str)': "La columna HORA tiene dos formatos distintos; se detecta cuál tiene cada registro",
'def hora_24(s):': "Convierte cualquiera de los dos formatos a hora 0-23",
'no_coincide = h24 != raw.TW.dt.hour': "¿La HORA escrita coincide con la hora de TW?",
'HORAS = pd.date_range("2017-01-01", "2019-12-31 23:00", freq="h")': "Todas las horas de 2017-2019: sirve para encontrar las horas que faltan en clima",
'pares_acc = set(zip(acc.BARRIO, acc.TW))': "Parejas (barrio, hora) positivas en cada tabla, para cruzarlas",
'inv = pd.DataFrame([': "Tabla 3: cada problema encontrado, cuántas filas afecta y su detalle",
# 1.6
'cl = clima[clima.TW < "2019-08-01"].merge(acc[["BARRIO", "TW"]].assign(y=1), on=["BARRIO", "TW"], how="left")':
    "Clima del periodo con etiqueta, marcado con y = 1 si hubo accidente en esa pareja",
'a_bruto = roc_auc_score(cl.y[ok], cl[v][ok])': "En bruto: ¿la variable sola separa horas con y sin accidente?",
'a_dentro = roc_auc_score(cl.y[ok], dentro)': "Dentro de barrio x hora: ¿separa comparando el mismo barrio a la misma hora?",
'filas.append(dict(variable=v, **{"AUC en bruto": max(a_bruto, 1 - a_bruto), "AUC dentro de barrio×hora": max(a_dentro, 1 - a_dentro)}))':
    "max(a, 1-a): interesa cuánto separa, sin importar el signo",
# 2
'BARRIOS = sorted(clima.BARRIO.unique())': "Rejilla completa 22 barrios x 26.280 horas: incluye las 440 parejas sin fila en clima",
'p = p.merge(clima, on=["BARRIO", "TW"], how="left")': "Se le une el clima (queda vacío en las horas sin clima)",
'p = p.merge(acc[["BARRIO", "TW"]].assign(target=1), on=["BARRIO", "TW"], how="left")': "Positivos = parejas de la tabla accidentes; negativos = todas las demás",
'n_acc = raw.groupby(["BARRIO", "TW"]).size().rename("n_acc")': "Número de accidentes por pareja (solo para describir el target y para los oráculos)",
'ent = p[p.target >= 0]': "Periodo con etiqueta (hasta 2019-07-31)",
'consulta = pd.read_sql("SELECT BARRIO, TW FROM clima WHERE TW >= \'2019-08-01 00:00:00\' AND TW < \'2019-12-31 00:00:00\'", con)':
    "Lista exacta de parejas a predecir, con la consulta del README",
'prev = ent.groupby(ent.TW.dt.to_period("M")).target.mean() * 100': "(a) Prevalencia mensual",
'k = ent[ent.target == 1].n_acc.clip(upper=4).value_counts(normalize=True).sort_index() * 100': "(b) Cuántos accidentes tienen las parejas positivas",
# 2.1
'rc = raw.copy()': "raw corregido: hora AM/PM arreglada y sin los 5 puntos fuera de la comuna",
'pares_corr = set(zip(rc.BARRIO, rc.TW))': "Se compara cuántas parejas positivas cambiarían frente a la tabla oficial",
# 3
'FESTIVOS = pd.to_datetime("""2017-01-01 2017-01-09 2017-03-20 2017-04-13 2017-04-14 2017-05-01 2017-05-29 2017-06-19 2017-06-26':
    "Festivos de Colombia 2017-2019 (incluye los trasladados a lunes por la Ley Emiliani y la Semana Santa)",
'def calendario(ts):': "Todas las variables salen de TW: se pueden calcular para cualquier fecha futura",
'p = pd.concat([p, calendario(p.TW)], axis=1)': "Se agregan las variables de calendario a toda la rejilla",
'desc = pd.DataFrame([': "Tabla 8: descripción de cada variable y por qué se creó",
'TIPO = np.select([ent.festivo == 1, ent.dow == 6, ent.dow == 5], ["festivo", "domingo", "sábado"], "laboral (lun-vie)")':
    "Tipo de día para la figura (el festivo tiene prioridad sobre el día de la semana)",
'perfil = ent.assign(tipo=TIPO).groupby(["tipo", "h"]).target.mean().unstack(0) * 100': "(a) % de horas-barrio con accidente por hora, para cada tipo de día",
'mes = ent.groupby("mes").target.mean() * 100': "(b) Por mes; en rojo diciembre y enero (variable dic_ene)",
'hh = np.arange(24)': "(c) Las 24 horas en el círculo (seno, coseno)",
# 4
'CORTES = ["2018-07-01", "2018-08-01", "2018-10-01", "2018-12-01", "2019-01-01", "2019-03-01"]':
    "Fechas de corte de la validación walk-forward; 2018-08 evalúa la misma temporada del leaderboard",
'def fold(c):': "Entrena con TODO lo anterior al corte (ventana expansiva) y valida en los 5 meses siguientes",
'filas = [(c[:7], pd.Timestamp(c), pd.Timestamp(c) + HORIZONTE, False) for c in CORTES] + [("entrega", CORTE, pd.Timestamp("2019-12-31"), True)]':
    "Figura 7: una fila por corte (entrenamiento + validación) y la entrega final",
'val = pd.DataFrame([dict(corte=c[:7], **{"entrena hasta": fold(c)[0].TW.max(), "valida": f"{fold(c)[1].TW.min():%Y-%m-%d} a {fold(c)[1].TW.max():%Y-%m-%d}",':
    "Tabla 9: verifica que cada corte entrena solo con el pasado y evalúa el periodo completo",
'VARIABLES = ["barrio", "h", "dow", "tipo_dia", "mes", "doy", "festivo", "vispera", "lunes_fest", "quincena", "dic_ene", "feria", "t"]':
    "Variables del modelo final: barrio + calendario + tendencia (ninguna depende de accidentes futuros)",
'PARAMS = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=400, subsample=0.8, subsample_freq=1,':
    "Hiperparámetros: árboles pequeños (15 hojas) y >= 400 filas por hoja (~7 accidentes por hoja con 1.8 % de positivos),\n"
    "tasa de aprendizaje baja con 400 árboles, submuestreo de filas y columnas y regularización L2: priorizan no memorizar\n"
    "ruido. Se fijaron en exploraciones previas; Optuna (40 intentos) no los mejoró (sección 5.3)",
'lgbm = lambda d, F=VARIABLES: lgb.LGBMClassifier(**PARAMS).fit(d[F], d.target)': "Entrena el LightGBM final con las variables F",
'tr_r, va_r = train_test_split(dev, test_size=0.2, random_state=SEED, stratify=dev.target)': "Partición aleatoria 80/20 sobre todo el periodo con etiqueta",
'tr_t, va_t = fold("2019-03-01")': "Partición temporal: entrena hasta feb-2019 y evalúa mar-jul 2019",
# 5.1
'ORACULOS = {"nivel real del día (22 barrios)": ["d"], "nivel real del día en ese barrio": ["d", "BARRIO"],':
    "Cada oráculo agrupa la validación de una forma y usa el número REAL de accidentes del grupo",
'base = tr.groupby(["BARRIO", "h"]).target.mean()': "Base: tasa barrio x hora por el perfil día x hora, ambos del entrenamiento",
'fila["+ " + nom] = roc_auc_score(va.target, va.base * (obs + 1) / (esp + 1))': "Se reescala la base por (observado / esperado) del grupo",
'ganancia = (orac.loc["promedio"] - orac.loc["promedio", "base: barrio × hora × día"]).drop("base: barrio × hora × día")':
    "Ganancia de cada oráculo frente a la base, promedio de los 6 cortes",
# 5.2
'for a_, b_ in [("BARRIO", "h"), ("BARRIO", "tipo_dia"), ("h", "tipo_dia")]:': "Interacciones como categorías, para la logística (one-hot)",
'y_lag = p.target.where(p.target >= 0).astype(float)': "Lag de 52 semanas: legal porque mira 364 días atrás y el horizonte es de 5 meses",
'CLIMA_F = ["precipIntensity", "temperature", "humidity", "windSpeed", "cloudCover", "visibility"]': "Variables de clima para la variante descartada",
'def tasa_bh(tr, va):': "M0: tasa histórica barrio x hora (el groupby del README)",
'def tasa_bht(tr, va, m=100):': "M1: tasa barrio x hora x tipo de día, suavizada hacia M0 con m filas de peso",
'def logistica(tr, va):': "M2: logística L2 sobre one-hot de calendario e interacciones",
'def con_lgbm(F):': "M3, M4 y variantes: LightGBM con el conjunto de variables F",
'PRED, filas = {}, []': "Se evalúan todos los modelos en los mismos 6 cortes; se guardan las predicciones para las secciones 6-9",
'orden = list(ESCALERA)[::-1]': "Figura 9: rango entre cortes (línea), cada corte (puntos pequeños) y promedio (punto grande)",
# 6
'CORTES_D = ["2018-08-01", "2019-03-01"]': "Dos cortes sin solapamiento: ago-dic 2018 (misma temporada) y mar-jul 2019",
'CAT_IDX = [i for i, v in enumerate(VARIABLES) if v not in ("doy", "t")]': "Para SMOTENC: todas las variables son categóricas salvo el día del año y la tendencia",
'def estrategia(nombre, tr):': "Entrena el modelo final con una técnica de balanceo aplicada SOLO al entrenamiento",
'd = pd.concat([pos, neg.iloc[rng.choice(len(neg), 9 * len(pos), replace=False)]])': "Quita negativos al azar hasta que los positivos sean el 10 %",
'd = pd.concat([neg, pos.iloc[rng.integers(0, len(pos), int(0.3 * len(neg) / 0.7))]])': "Replica positivos con reemplazo hasta que sean el 30 %",
'X = tr[VARIABLES].assign(barrio=tr.barrio.cat.codes)': "SMOTENC necesita números: el barrio pasa a código y luego vuelve a categoría",
's = estrategia(e, tr).predict_proba(va[VARIABLES])[:, 1]': "La validación nunca se remuestrea: se evalúa con su desbalance real",
'des_m = des.groupby("estrategia", sort=False)[["ROC-AUC", "PR-AUC", "prob. media predicha", "prevalencia real"]].mean()': "Tabla 13: promedio de los dos cortes",
# 6.1
'P18, P19 = PRED["M4 · LightGBM: barrio + calendario + t (FINAL)", "2018-08-01"], PRED["M4 · LightGBM: barrio + calendario + t (FINAL)", "2019-03-01"]':
    "Predicciones del modelo final en los dos cortes (fuera de muestra)",
'def costo(y, s, umbral, k):': "Costo total con C_FP = 1 (patrulla sin accidente) y C_FN = k (accidente sin patrulla)",
'grid = np.quantile(P18, np.linspace(0, 0.999, 400))': "Umbrales candidatos: 400 cuantiles de las probabilidades de 2018-08",
'sin_patrulla = costo(Y18, P18, 2, k)[0]': "Referencia: no patrullar nunca (umbral imposible = 2)",
'UMBRAL[k] = grid[cs.argmin()]': "Umbral de mínimo costo, elegido SOLO con 2018-08",
'for nom, y, s in [("2018-08 (elige)", Y18, P18), ("2019-03 (aplica)", Y19, P19)]:': "Se aplica el mismo umbral, sin cambiarlo, al corte 2019-03",
# 7
'U = UMBRAL[25]': "Umbral del escenario elegido (C_FN / C_FP = 25)",
'cm = confusion_matrix(y, s >= U)': "Matriz de confusión en el umbral elegido",
# 8
'tr_f = p[(p.TW < CORTE) & (p.target >= 0)]': "Entrenamiento final: todo lo etiquetado hasta 2019-07-31",
'te_f = p[(p.TW >= CORTE) & (p.TW <= "2019-12-30 23:00")].copy()': "Periodo a predecir: 2019-08-01 00:00 a 2019-12-30 23:00",
'modelo_final = lgbm(tr_f)': "Mismo modelo M4, mismos hiperparámetros y misma semilla que en la validación",
'SUBIDO = "mi_prediccion_Savelezc.csv"                                      # archivo que se subió al leaderboard':
    "Si el archivo subido está en la carpeta, se comprueba que este notebook lo reproduce",
'REF = {"tasa histórica barrio × hora (README)": 0.7589, "baseline del profesor (README)": 0.7739}': "Referencias del README",
'rng = np.random.default_rng(SEED)': "Rango por azar: 1.000 remuestreos de días completos del corte 2018-08 (misma temporada)",
'imp = pd.Series(modelo_final.booster_.feature_importance("gain"), index=VARIABLES)': "(b) Importancia: % de la ganancia de los splits del modelo final",
# 9
'te_f["tipo"] = np.where((te_f.festivo == 1) | (te_f.dow >= 5), "fin de semana o festivo", "laboral")':
    "Figura 14: riesgo predicho promedio por barrio y hora, separado por tipo de día",
'orden_b = te_f.groupby("BARRIO").prob.mean().sort_values(ascending=False).index': "Barrios ordenados de mayor a menor riesgo",
'va19 = fold("2019-03-01")[1].assign(prob=P19)': "Patrullas: cada día se cubren las 4·N parejas barrio-hora de mayor probabilidad (corte 2019-03)",
'cub = va19.rk <= 4 * n': "Parejas cubiertas con N patrullas de 4 horas",
'dia = te_f[te_f.TW.dt.normalize() == "2019-08-05"].nlargest(12, "prob")[["TW", "BARRIO", "prob"]]': "Ejemplo de turno para un día concreto del periodo a predecir",
}
src = open(sys.argv[1]).read().split("\n")
out, usados = [], set()
for line in src:
    s = line.strip()
    if s in C:
        ind = line[: len(line) - len(line.lstrip())]
        out += [ind + "# " + t for t in C[s].split("\n")]
        usados.add(s)
    out.append(line)
faltan = [k for k in C if k not in usados]
print("comentarios insertados:", len(usados), "| sin encontrar:", faltan)
open(sys.argv[1], "w").write("\n".join(out))
