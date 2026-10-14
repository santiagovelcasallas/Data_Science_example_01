"""Genera Reto2_Accidentalidad_Poblado.ipynb. Las celdas de codigo estan aqui; los markdown explicativos estan en
notebook_md.py y se escriben despues de leer las salidas reales del notebook ejecutado."""
import nbformat as nbf
from notebook_md import MD

C = []          # lista de (tipo, texto)
md = lambda k: C.append(("md", MD[k]))
code = lambda s: C.append(("code", s.strip("\n")))

md("portada")
md("entorno")
code(r'''
import importlib.util, subprocess, sys, shutil

REQUERIDAS = {"lightgbm": "lightgbm", "seaborn": "seaborn", "imblearn": "imbalanced-learn"}
faltan = [pkg for mod, pkg in REQUERIDAS.items() if importlib.util.find_spec(mod) is None]
if faltan:
    print("Instalando:", ", ".join(faltan))
    if importlib.util.find_spec("pip") is not None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *faltan])
    elif shutil.which("uv"):
        subprocess.check_call(["uv", "pip", "install", "--python", sys.executable, *faltan])
    else:
        raise RuntimeError(f"Instale manualmente: {' '.join(faltan)}")
    importlib.invalidate_caches()
print("Dependencias listas.")
''')
code(r'''
import os, sqlite3, urllib.request, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import lightgbm as lgb
from IPython.display import display
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

warnings.filterwarnings("ignore")
SEED = 0
AZUL, NARANJA, VERDE, AMARILLO, VIOLETA, ROJO, GRIS = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e34948", "#8a8984"
sns.set_theme(style="whitegrid", context="notebook", palette=[AZUL, NARANJA, VERDE, AMARILLO, VIOLETA, ROJO])
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False})
pd.set_option("display.max_colwidth", 120)

def tabla(df, n, titulo):
    print(f"Tabla {n} — {titulo}")
    display(df)
''')
code(r'''
URL = ("https://d3qixogk4zgixq.cloudfront.net/data/prediccion-accidentalidad-poblado/"
       "data_accidentes_poblado.sqlite3")
DB = "data_accidentes_poblado.sqlite3"
if not os.path.exists(DB):
    urllib.request.urlretrieve(URL, DB)
con = sqlite3.connect(DB)
clima = pd.read_sql("SELECT * FROM clima", con)
acc = pd.read_sql("SELECT * FROM accidentes", con)
raw = pd.read_sql("SELECT * FROM raw_accidentes", con)
print(f"clima {clima.shape} | accidentes {acc.shape} | raw_accidentes {raw.shape}")
''')

# ---------------------------------------------------------------- 1. EDA
md("s1")
code(r'''
resumen = pd.DataFrame({
    "filas": [len(clima), len(acc), len(raw)],
    "columnas": [clima.shape[1], acc.shape[1], raw.shape[1]],
    "TW mínimo": [clima.TW.min(), acc.TW.min(), raw.TW.min()],
    "TW máximo": [clima.TW.max(), acc.TW.max(), raw.TW.max()],
    "barrios": [clima.BARRIO.nunique(), acc.BARRIO.nunique(), raw.BARRIO.nunique()],
    "celdas faltantes": [int(clima.isna().sum().sum()), int(acc.isna().sum().sum()), int(raw.isna().sum().sum())],
}, index=["clima", "accidentes", "raw_accidentes"])
tabla(resumen, 1, "Las tres tablas del SQLite")
print("Mismos 22 barrios, escritos igual, en las tres tablas:", set(clima.BARRIO) == set(acc.BARRIO) == set(raw.BARRIO))
print("Pares (barrio, hora) duplicados — clima:", int(clima.duplicated(["BARRIO", "TW"]).sum()),
      "| accidentes:", int(acc.duplicated(["BARRIO", "TW"]).sum()))
''')
md("s1_1")
code(r'''
for d in (clima, acc, raw):
    d["TW"] = pd.to_datetime(d.TW)
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]

fig, ax = plt.subplots(1, 2, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
m = raw.groupby(raw.TW.dt.to_period("M")).size()
m.index = m.index.to_timestamp()
ax[0].plot(m.index, m.values, color=AZUL, lw=2, marker="o", ms=3)
ax[0].axhline(m.mean(), color=GRIS, ls="--", lw=1)
ax[0].text(m.index[0], m.mean() + 6, f"media {m.mean():.0f}/mes", color=GRIS, fontsize=9)
ax[0].set(title="(a) Accidentes por mes (raw_accidentes)", ylabel="accidentes", xlabel="")
hm = acc.groupby([acc.TW.dt.dayofweek, acc.TW.dt.hour]).size().unstack(fill_value=0)
hm.index = DIAS
sns.heatmap(hm, cmap="Blues", ax=ax[1], cbar_kws=dict(label="horas-barrio con accidente"))
ax[1].set(title="(b) Horas-barrio con accidente, por día y hora", xlabel="hora del día", ylabel="")
fig.suptitle("Figura 1 — Distribución temporal de los accidentes (ene-2017 a jul-2019)", y=1.02)
plt.tight_layout(); plt.show()
por_ano = raw.groupby(raw.TW.dt.year).size()
print("Accidentes por año:", por_ano.to_dict(), "(2019 solo hasta julio)")
print("Mes mínimo:", m.idxmin().strftime("%Y-%m"), int(m.min()), "| mes máximo:", m.idxmax().strftime("%Y-%m"), int(m.max()))
''')
md("s1_2")
code(r'''
fuera = raw.Lon < -75.62
fig, ax = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw=dict(width_ratios=[1, 1.1, 0.8]))
cnt = raw.BARRIO.value_counts().sort_values()
ax[0].barh(cnt.index, cnt.values, color=AZUL)
ax[0].bar_label(ax[0].containers[0], padding=2, fontsize=8)
ax[0].set(title="(a) Accidentes por barrio", xlabel="accidentes")
ax[1].scatter(raw.Lon[~fuera], raw.Lat[~fuera], s=3, alpha=.25, color=AZUL)
cent = raw[~fuera].groupby("BARRIO")[["Lon", "Lat"]].median()
for b, r in cent.iterrows():
    ax[1].annotate(b, (r.Lon, r.Lat), fontsize=7, ha="center", color="#222",
                   bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=.7))
ax[1].set(title="(b) Ubicación de cada accidente en El Poblado", xlabel="longitud", ylabel="latitud")
ax[2].scatter(raw.Lon[~fuera], raw.Lat[~fuera], s=3, alpha=.25, color=AZUL, label="El Poblado")
ax[2].scatter(raw.Lon[fuera], raw.Lat[fuera], s=70, color=ROJO, marker="x", label=f"'laflorida' fuera de la comuna ({fuera.sum()})")
ax[2].set(title="(c) Vista completa: 5 puntos lejos de la comuna", xlabel="longitud")
ax[2].legend(fontsize=8, loc="lower right")
fig.suptitle("Figura 2 — Distribución espacial de los accidentes", y=1.01)
plt.tight_layout(); plt.show()
print(f"Barrio con más accidentes: {cnt.index[-1]} ({cnt.iloc[-1]}); con menos: {cnt.index[0]} ({cnt.iloc[0]}). "
      f"Razón máx/mín = {cnt.iloc[-1] / cnt.iloc[0]:.0f}x")
''')
md("s1_3")
code(r'''
VARS_CLIMA = ["precipIntensity", "precipProbability", "summary", "temperature", "apparentTemperature", "dewPoint",
              "humidity", "windSpeed", "windBearing", "cloudCover", "uvIndex", "visibility"]
falt = (clima.groupby(clima.TW.dt.to_period("M"))[VARS_CLIMA].apply(lambda g: g.isna().mean() * 100)).T
falt.columns = [c.strftime("%Y-%m") for c in falt.columns]
fig, ax = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw=dict(width_ratios=[2.2, 1]))
sns.heatmap(falt, cmap="Reds", vmin=0, vmax=100, ax=ax[0], cbar_kws=dict(label="% de filas faltantes"))
ax[0].set(title="(a) % de faltantes por variable y mes", xlabel="", ylabel="")
sns.boxplot(x=clima.TW.dt.hour, y=clima.temperature, color=AZUL, fliersize=0, linewidth=.8, ax=ax[1])
ax[1].set(title="(b) Temperatura según la hora del día", xlabel="hora", ylabel="°C")
ax[1].set_xticks(range(0, 24, 3))
fig.suptitle("Figura 3 — Calidad y comportamiento del clima", y=1.02)
plt.tight_layout(); plt.show()

nan = clima[VARS_CLIMA].isna().sum()
calma = (clima.windBearing.isna() & (clima.windSpeed == 0)).sum()
t2 = pd.DataFrame({"faltantes": nan, "% del total": (100 * nan / len(clima)).round(2)}).sort_values("faltantes", ascending=False)
tabla(t2, 2, "Faltantes por variable de clima")
bloque = clima[clima.precipIntensity.isna()].TW
print(f"precipIntensity faltante: {bloque.dt.floor('h').nunique()} horas distintas, "
      f"entre {bloque.min():%Y-%m-%d} y {bloque.max():%Y-%m-%d}; "
      f"en esas horas falta en {clima[clima.TW.isin(bloque)].groupby('TW').precipIntensity.apply(lambda s: s.isna().mean()).mean():.0%} de los barrios")
print(f"windBearing faltante con windSpeed = 0 (viento en calma): {calma:,} de {int(nan['windBearing']):,}")
print("Correlación de la temperatura entre barrios a la misma hora:",
      round(clima.pivot_table(index="TW", columns="BARRIO", values="temperature").corr().values[np.triu_indices(22, 1)].mean(), 3))
''')
md("s1_4")
code(r'''
h = raw.HORA.astype(str)
fmt_ampm = h.str.match(r"^\d{1,2}:\d{2} (AM|PM)$")
fmt_largo = h.str.match(r"^\d{1,2}:\d{2}:\d{2} (a|p)\. m\.$")

def hora_24(s):
    hh, pm = int(s.split(":")[0]), ("PM" in s) or ("p. m." in s)
    return hh % 12 + (12 if pm else 0)

h24 = h.map(hora_24)
no_coincide = h24 != raw.TW.dt.hour
HORAS = pd.date_range("2017-01-01", "2019-12-31 23:00", freq="h")
horas_sin_clima = HORAS.difference(clima.TW.unique())
pares_acc = set(zip(acc.BARRIO, acc.TW))
pares_raw = set(zip(raw.BARRIO, raw.TW))

inv = pd.DataFrame([
    ["raw: HORA en formato «hh:mm AM/PM»", fmt_ampm.sum(), f"hasta {raw.TW[fmt_ampm].max():%Y-%m-%d}"],
    ["raw: HORA en formato «hh:mm:ss a. m./p. m.»", fmt_largo.sum(), f"desde {raw.TW[fmt_largo].min():%Y-%m-%d}"],
    ["raw: HORA no coincide con la hora de TW", no_coincide.sum(), "error AM/PM (Tabla 4)"],
    ["raw: 'laflorida' geocodificado en San Antonio de Prado", fuera.sum(), "COMUNA = Corregimiento de San Antonio de Prado"],
    ["raw: COMUNA con el nombre de un barrio", (~raw.COMUNA.isin(["El Poblado", "Corregimiento de San Antonio de Prado"])).sum(), "coordenadas dentro de El Poblado"],
    ["raw: CLASE «Caída Ocupante» vs «Caida Ocupante»", (raw.CLASE == "Caída Ocupante").sum(), "misma categoría escrita distinto"],
    ["raw: RADICADO con sufijo «.0»", raw.RADICADO.astype(str).str.endswith(".0").sum(), "mezcla entero y texto"],
    ["raw: DISENO faltante", raw.DISENO.isna().sum(), ""],
    ["accidentes vs raw: pares (barrio, hora) distintos", len(pares_acc ^ pares_raw), f"{len(pares_acc):,} pares en ambas"],
    ["clima: horas sin fila (para los 22 barrios)", len(horas_sin_clima), f"{22 * len(horas_sin_clima)} filas; {acc.TW.isin(horas_sin_clima).sum()} con accidente"],
    ["clima: precipIntensity faltante", clima.precipIntensity.isna().sum(), "casi todo entre may y ago de 2018 (Figura 3a)"],
    ["clima: windBearing faltante", clima.windBearing.isna().sum(), f"{calma:,} con viento en calma"],
], columns=["problema", "filas", "detalle"])
tabla(inv, 3, "Inventario de inconsistencias")
tabla(raw.loc[no_coincide, ["RADICADO", "FECHA", "HORA", "TW", "BARRIO", "CLASE"]], 4, "Registros cuya HORA no coincide con TW")
''')
md("s1_5")
code(r'''
cl = clima[clima.TW < "2019-08-01"].merge(acc[["BARRIO", "TW"]].assign(y=1), on=["BARRIO", "TW"], how="left")
cl["y"] = cl.y.fillna(0)
cl["h"] = cl.TW.dt.hour
NUM_CLIMA = ["temperature", "apparentTemperature", "dewPoint", "humidity", "uvIndex", "visibility", "cloudCover",
             "windSpeed", "precipIntensity", "precipProbability"]
filas = []
for v in NUM_CLIMA:
    ok = cl[v].notna()
    a_bruto = roc_auc_score(cl.y[ok], cl[v][ok])
    dentro = cl[ok].groupby(["BARRIO", "h"])[v].rank(pct=True)          # posición del valor dentro de su barrio x hora
    a_dentro = roc_auc_score(cl.y[ok], dentro)
    filas.append(dict(variable=v, **{"AUC en bruto": max(a_bruto, 1 - a_bruto), "AUC dentro de barrio×hora": max(a_dentro, 1 - a_dentro)}))
auc_clima = pd.DataFrame(filas).set_index("variable").sort_values("AUC en bruto")

fig, ax = plt.subplots(figsize=(10, 5))
auc_clima.plot.barh(ax=ax, color=[GRIS, AZUL], width=.75)
ax.axvline(0.5, color=ROJO, ls="--", lw=1); ax.text(0.502, -0.9, "azar", color=ROJO, fontsize=9)
ax.set(xlim=(0.45, 0.72), xlabel="ROC-AUC de la variable sola", ylabel="",
       title="Figura 4 — ¿Cuánto separa cada variable de clima las horas con y sin accidente?")
ax.legend(loc="lower right"); plt.tight_layout(); plt.show()
tabla(auc_clima.round(3), 5, "ROC-AUC de cada variable de clima, en bruto y dentro de la misma pareja barrio × hora")
''')

# ---------------------------------------------------------------- 2. Target
md("s2")
code(r'''
BARRIOS = sorted(clima.BARRIO.unique())
p = pd.MultiIndex.from_product([BARRIOS, HORAS], names=["BARRIO", "TW"]).to_frame(index=False)
p = p.merge(clima, on=["BARRIO", "TW"], how="left")
p = p.merge(acc[["BARRIO", "TW"]].assign(target=1), on=["BARRIO", "TW"], how="left")
p["target"] = p.target.fillna(0).astype("int8")
CORTE = pd.Timestamp("2019-08-01")
p.loc[p.TW >= CORTE, "target"] = -1                                    # periodo de evaluación: etiqueta oculta
n_acc = raw.groupby(["BARRIO", "TW"]).size().rename("n_acc")
p = p.join(n_acc, on=["BARRIO", "TW"])
p["n_acc"] = p.n_acc.fillna(0).astype(int)

ent = p[p.target >= 0]
consulta = pd.read_sql("SELECT BARRIO, TW FROM clima WHERE TW >= '2019-08-01 00:00:00' AND TW < '2019-12-31 00:00:00'", con)
universo = pd.Series({
    "filas de la rejilla barrio × hora (2017-2019)": len(p),
    "  · con fila en clima": len(p) - 22 * len(horas_sin_clima),
    "  · agregadas (horas sin clima)": 22 * len(horas_sin_clima),
    "entrenamiento (hasta 2019-07-31)": len(ent),
    "  · positivos (≥ 1 accidente)": int(ent.target.sum()),
    "  · prevalencia": f"{ent.target.mean():.2%}",
    "evaluación (consulta del README)": len(consulta),
})
tabla(universo.to_frame("valor"), 6, "Unidad de análisis y tamaño del problema")

fig, ax = plt.subplots(1, 2, figsize=(16, 4.5), gridspec_kw=dict(width_ratios=[1.5, 1]))
prev = ent.groupby(ent.TW.dt.to_period("M")).target.mean() * 100
prev.index = prev.index.to_timestamp()
ax[0].plot(prev.index, prev.values, color=AZUL, lw=2, marker="o", ms=3)
ax[0].axhline(ent.target.mean() * 100, color=GRIS, ls="--", lw=1)
ax[0].set(title="(a) Prevalencia mensual: % de parejas barrio-hora con accidente", ylabel="%", xlabel="")
k = ent[ent.target == 1].n_acc.clip(upper=4).value_counts(normalize=True).sort_index() * 100
k.index = ["1", "2", "3", "4+"][: len(k)]
b = ax[1].bar(k.index, k.values, color=AZUL); ax[1].bar_label(b, fmt="%.1f%%")
ax[1].set(title="(b) Accidentes en las horas-barrio positivas", xlabel="accidentes en la misma hora y barrio", ylabel="% de positivos")
fig.suptitle("Figura 5 — El target: presencia de al menos un accidente", y=1.02)
plt.tight_layout(); plt.show()
''')
md("s2_1")
code(r'''
rc = raw.copy()
rc.loc[no_coincide, "TW"] = rc.loc[no_coincide, "TW"].dt.normalize() + pd.to_timedelta(h24[no_coincide], unit="h")
rc = rc[~fuera]
pares_corr = set(zip(rc.BARRIO, rc.TW))
efecto = pd.Series({
    "pares positivos en la tabla accidentes (oficial)": len(pares_acc),
    "pares positivos si se corrige raw": len(pares_corr),
    "pares que dejarían de ser positivos": len(pares_acc - pares_corr),
    "pares nuevos positivos": len(pares_corr - pares_acc),
    "% de positivos afectados": f"{100 * len(pares_acc ^ pares_corr) / len(pares_acc):.2f} %",
})
tabla(efecto.to_frame("valor"), 7, "Efecto de corregir raw_accidentes sobre el target")
''')

# ---------------------------------------------------------------- 3. Variables
md("s3")
code(r'''
FESTIVOS = pd.to_datetime("""2017-01-01 2017-01-09 2017-03-20 2017-04-13 2017-04-14 2017-05-01 2017-05-29 2017-06-19 2017-06-26
2017-07-03 2017-07-20 2017-08-07 2017-08-21 2017-10-16 2017-11-06 2017-11-13 2017-12-08 2017-12-25
2018-01-01 2018-01-08 2018-03-19 2018-03-29 2018-03-30 2018-05-01 2018-05-14 2018-06-04 2018-06-11 2018-07-02
2018-07-20 2018-08-07 2018-08-20 2018-10-15 2018-11-05 2018-11-12 2018-12-08 2018-12-25
2019-01-01 2019-01-07 2019-03-25 2019-04-18 2019-04-19 2019-05-01 2019-06-03 2019-06-24 2019-07-01 2019-07-20
2019-08-07 2019-08-19 2019-10-14 2019-11-04 2019-11-11 2019-12-08 2019-12-25""".split())

def calendario(ts):
    d = ts.dt.normalize()
    out = pd.DataFrame(index=ts.index)
    out["h"] = ts.dt.hour; out["dow"] = ts.dt.dayofweek; out["mes"] = ts.dt.month
    out["doy"] = ts.dt.dayofyear; out["dom"] = ts.dt.day
    fest = d.isin(FESTIVOS)
    out["festivo"] = fest.astype(int)
    out["tipo_dia"] = np.where(fest, 6, out.dow)                          # un festivo se comporta como domingo
    out["vispera"] = (d + pd.Timedelta(days=1)).isin(FESTIVOS).astype(int)
    out["lunes_fest"] = (fest & (out.dow == 0)).astype(int)               # puente (Ley Emiliani)
    out["quincena"] = out.dom.isin([14, 15, 16, 29, 30, 31, 1]).astype(int)
    out["dic_ene"] = (((out.mes == 12) & (out.dom >= 15)) | ((out.mes == 1) & (out.dom <= 10))).astype(int)
    out["feria"] = ((out.mes == 8) & (out.dom <= 12)).astype(int)         # Feria de las Flores
    out["t"] = (ts - pd.Timestamp("2017-01-01")).dt.days / 365.0          # tendencia, en años
    out["hs"] = np.sin(2 * np.pi * out.h / 24); out["hc"] = np.cos(2 * np.pi * out.h / 24)
    out["ys"] = np.sin(2 * np.pi * out.doy / 365.25); out["yc"] = np.cos(2 * np.pi * out.doy / 365.25)
    return out

p = pd.concat([p, calendario(p.TW)], axis=1)
p["barrio"] = p.BARRIO.astype("category")                                # categórica nativa de LightGBM
ent = p[p.target >= 0]
print(f"Festivos en la lista: {len(FESTIVOS)} ({FESTIVOS.year.value_counts().sort_index().to_dict()})")

desc = pd.DataFrame([
    ["barrio", "categórica (22)", "nombre del barrio", "los accidentes varían 14 veces entre barrios (Figura 2a)"],
    ["h, dow, mes, doy", "enteras", "hora, día de la semana, mes y día del año de TW", "el perfil horario domina (Figura 1b)"],
    ["festivo, tipo_dia", "binaria / 0-6", "festivo de Colombia; tipo_dia = día de semana, con festivo = 6", "un festivo apaga la hora pico (Figura 6a)"],
    ["vispera, lunes_fest", "binarias", "víspera de festivo; lunes festivo (puente)", "cambian la movilidad del día"],
    ["quincena", "binaria", "días 14-16 y 29-1", "días de pago"],
    ["dic_ene, feria", "binarias", "15-dic a 10-ene; 1-12 de agosto", "temporadas especiales del calendario"],
    ["t", "continua", "años desde 2017-01-01", "deja ver el cambio lento de cada barrio"],
    ["hs, hc, ys, yc", "continuas", "seno y coseno de la hora y del día del año", "codificación cíclica (se compara en 5.2)"],
], columns=["variable", "tipo", "cómo se calcula", "por qué"])
tabla(desc, 8, "Variables creadas a partir de TW")
''')
md("s3_1")
code(r'''
TIPO = np.select([ent.festivo == 1, ent.dow == 6, ent.dow == 5], ["festivo", "domingo", "sábado"], "laboral (lun-vie)")
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1, 0.75]))
perfil = ent.assign(tipo=TIPO).groupby(["tipo", "h"]).target.mean().unstack(0) * 100
for col, c in zip(["laboral (lun-vie)", "sábado", "domingo", "festivo"], [AZUL, NARANJA, VERDE, ROJO]):
    ax[0].plot(perfil.index, perfil[col], color=c, lw=2, label=col)
ax[0].set(title="(a) % de horas-barrio con accidente, por hora y tipo de día", xlabel="hora", ylabel="%")
ax[0].set_xticks(range(0, 24, 2)); ax[0].legend()
mes = ent.groupby("mes").target.mean() * 100
b = ax[1].bar(mes.index, mes.values, color=[ROJO if i in (1, 12) else AZUL for i in mes.index])
ax[1].set(title="(b) % de horas-barrio con accidente, por mes", xlabel="mes", ylabel="%", ylim=(mes.min() * .8, mes.max() * 1.08))
ax[1].set_xticks(range(1, 13))
hh = np.arange(24)
sc = ax[2].scatter(np.sin(2 * np.pi * hh / 24), np.cos(2 * np.pi * hh / 24), c=hh, cmap="twilight", s=90)
for i in (0, 6, 12, 18, 23):
    ax[2].annotate(f"{i}h", (np.sin(2 * np.pi * i / 24) * 1.15, np.cos(2 * np.pi * i / 24) * 1.15), ha="center", va="center", fontsize=9)
ax[2].set(title="(c) Codificación cíclica de la hora", xlabel="hs = sin(2πh/24)", ylabel="hc = cos(2πh/24)", aspect="equal",
          xlim=(-1.35, 1.35), ylim=(-1.35, 1.35))
fig.suptitle("Figura 6 — Por qué estas variables de calendario", y=1.02)
plt.tight_layout(); plt.show()
print("Hora pico laboral:", perfil["laboral (lun-vie)"].idxmax(), "h con", round(perfil["laboral (lun-vie)"].max(), 2),
      "% | misma hora en festivo:", round(perfil.loc[perfil["laboral (lun-vie)"].idxmax(), "festivo"], 2), "%")
print("A las 8 h: laboral", round(perfil.loc[8, "laboral (lun-vie)"], 2), "% vs festivo", round(perfil.loc[8, "festivo"], 2), "%")
''')

# ---------------------------------------------------------------- 4. Validacion
md("s4")
code(r'''
CORTES = ["2018-07-01", "2018-08-01", "2018-10-01", "2018-12-01", "2019-01-01", "2019-03-01"]
HORIZONTE = pd.DateOffset(months=5)                                       # igual que ago-dic 2019

def fold(c):
    a = pd.Timestamp(c)
    return p[(p.TW < a) & (p.target >= 0)], p[(p.TW >= a) & (p.TW < a + HORIZONTE)]

fig, ax = plt.subplots(figsize=(15, 4.3))
filas = [(c[:7], pd.Timestamp(c), pd.Timestamp(c) + HORIZONTE, False) for c in CORTES] + [("entrega", CORTE, pd.Timestamp("2019-12-31"), True)]
for i, (nom, a, b, oculto) in enumerate(filas):
    ax.barh(i, mdates.date2num(a) - mdates.date2num(pd.Timestamp("2017-01-01")), left=mdates.date2num(pd.Timestamp("2017-01-01")),
            color=AZUL, height=.6, label="entrenamiento" if i == 0 else None)
    ax.barh(i, mdates.date2num(b) - mdates.date2num(a), left=mdates.date2num(a), height=.6,
            color=GRIS if oculto else NARANJA, hatch="//" if oculto else None, edgecolor="white",
            label=("evaluación del leaderboard (etiquetas ocultas)" if oculto else ("validación (5 meses)" if i == 0 else None)))
ax.set_yticks(range(len(filas)), [f"corte {f[0]}" if not f[3] else "entrega final" for f in filas])
ax.invert_yaxis(); ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9, frameon=False)
ax.set(title="Figura 7 — Validación walk-forward con ventana expansiva: 6 cortes de 5 meses + la entrega")
plt.tight_layout(); plt.show()

val = pd.DataFrame([dict(corte=c[:7], **{"entrena hasta": fold(c)[0].TW.max(), "valida": f"{fold(c)[1].TW.min():%Y-%m-%d} a {fold(c)[1].TW.max():%Y-%m-%d}",
                                         "filas validación": len(fold(c)[1]), "positivos": int(fold(c)[1].target.sum())}) for c in CORTES])
tabla(val, 9, "Los 6 cortes de validación (cada uno evalúa todas las parejas barrio × hora del periodo)")
''')
md("s4_1")
code(r'''
VARIABLES = ["barrio", "h", "dow", "tipo_dia", "mes", "doy", "festivo", "vispera", "lunes_fest", "quincena", "dic_ene", "feria", "t"]
PARAMS = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=400, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5, random_state=SEED, verbose=-1)
lgbm = lambda d, F=VARIABLES: lgb.LGBMClassifier(**PARAMS).fit(d[F], d.target)

dev = p[p.target >= 0]
tr_r, va_r = train_test_split(dev, test_size=0.2, random_state=SEED, stratify=dev.target)
auc_aleat = roc_auc_score(va_r.target, lgbm(tr_r).predict_proba(va_r[VARIABLES])[:, 1])
tr_t, va_t = fold("2019-03-01")
auc_temp = roc_auc_score(va_t.target, lgbm(tr_t).predict_proba(va_t[VARIABLES])[:, 1])
tabla(pd.DataFrame({"ROC-AUC": [auc_aleat, auc_temp], "¿el modelo vio el futuro?": ["sí: entrena con días posteriores a los que evalúa", "no"]},
                   index=["partición aleatoria 80/20 (todo 2017 a jul-2019)", "partición temporal (corte 2019-03)"]).round(4),
      10, "Mismo modelo, dos formas de partir los datos")
''')
md("s4_2")

# ---------------------------------------------------------------- 5. Modelado
code(r'''
ORACULOS = {"nivel real del día (22 barrios)": ["d"], "nivel real del día en ese barrio": ["d", "BARRIO"],
            "nivel real de la semana": ["sem"], "nivel real del mes en ese barrio": ["mes", "BARRIO"],
            "nivel real de esa hora (22 barrios)": ["TW"]}
filas = []
for c in CORTES:
    tr, va = fold(c)
    va = va.assign(d=va.TW.dt.normalize(), sem=va.TW.dt.to_period("W").astype(str))
    base = tr.groupby(["BARRIO", "h"]).target.mean()
    prof = tr.groupby(["dow", "h"]).target.mean() / tr.target.mean()
    va["base"] = va.set_index(["BARRIO", "h"]).index.map(base).values * va.set_index(["dow", "h"]).index.map(prof).values
    fila = {"corte": c[:7], "base: barrio × hora × día": roc_auc_score(va.target, va.base)}
    for nom, key in ORACULOS.items():
        obs = va.groupby(key).n_acc.transform("sum") - va.n_acc           # se excluye la propia fila
        esp = va.groupby(key).base.transform("sum") - va.base
        fila["+ " + nom] = roc_auc_score(va.target, va.base * (obs + 1) / (esp + 1))
    filas.append(fila)
orac = pd.DataFrame(filas).set_index("corte")
orac.loc["promedio"] = orac.mean()
ganancia = (orac.loc["promedio"] - orac.loc["promedio", "base: barrio × hora × día"]).drop("base: barrio × hora × día")

fig, ax = plt.subplots(figsize=(10, 4))
b = ax.barh(ganancia.index[::-1], ganancia.values[::-1], color=[ROJO if g < 0 else AZUL for g in ganancia.values[::-1]])
ax.bar_label(b, fmt="%+.4f", padding=3); ax.axvline(0, color="#333", lw=1)
ax.set(xlabel="cambio en ROC-AUC frente a la base (promedio de 6 cortes)", xlim=(min(ganancia.min() * 1.6, -0.003), 0.004),
       title="Figura 8 — Oráculos: ¿ayuda conocer el nivel REAL de accidentes del día, la semana o el mes?")
plt.tight_layout(); plt.show()
tabla(orac.round(4), 11, "ROC-AUC de la base y de la base ajustada con niveles reales (sin la propia fila)")
''')
md("s5_1")
code(r'''
for a_, b_ in [("BARRIO", "h"), ("BARRIO", "tipo_dia"), ("h", "tipo_dia")]:
    p[f"{a_}_{b_}"] = p[a_].astype(str) + "_" + p[b_].astype(str)
y_lag = p.target.where(p.target >= 0).astype(float)
p["lag52"] = y_lag.groupby(p.BARRIO).shift(52 * 7 * 24)                  # misma hora y día de la semana, 52 semanas antes
CLIMA_F = ["precipIntensity", "temperature", "humidity", "windSpeed", "cloudCover", "visibility"]
LOG_F = ["BARRIO", "h", "tipo_dia", "mes", "festivo", "vispera", "lunes_fest", "quincena", "dic_ene", "feria",
         "BARRIO_h", "BARRIO_tipo_dia", "h_tipo_dia"]

def tasa_bh(tr, va):
    return va.set_index(["BARRIO", "h"]).index.map(tr.groupby(["BARRIO", "h"]).target.mean()).values

def tasa_bht(tr, va, m=100):
    prior = tasa_bh(tr, va)
    st = tr.groupby(["BARRIO", "h", "tipo_dia"]).target.agg(["sum", "count"])
    k = pd.MultiIndex.from_frame(va[["BARRIO", "h", "tipo_dia"]])
    return (st["sum"].reindex(k).fillna(0).values + m * prior) / (st["count"].reindex(k).fillna(0).values + m)

def logistica(tr, va):
    enc = OneHotEncoder(handle_unknown="ignore").fit(tr[LOG_F].astype(str))
    m = LogisticRegression(C=0.03, max_iter=2000).fit(enc.transform(tr[LOG_F].astype(str)), tr.target)
    return m.predict_proba(enc.transform(va[LOG_F].astype(str)))[:, 1]

def con_lgbm(F):
    return lambda tr, va: lgbm(tr, F).predict_proba(va[F])[:, 1]

ESCALERA = {
    "M0 · tasa histórica barrio × hora": tasa_bh,
    "M1 · tasa barrio × hora × tipo de día": tasa_bht,
    "M2 · logística L2 con interacciones": logistica,
    "M3 · LightGBM: barrio + calendario": con_lgbm([v for v in VARIABLES if v != "t"]),
    "M4 · LightGBM: barrio + calendario + t (FINAL)": con_lgbm(VARIABLES),
    "M4 + clima": con_lgbm(VARIABLES + CLIMA_F),
    "M4 + lag de 52 semanas": con_lgbm(VARIABLES + ["lag52"]),
    "M4 + codificación cíclica": con_lgbm(VARIABLES + ["hs", "hc", "ys", "yc"]),
}
PRED, filas = {}, []
for c in CORTES:
    tr, va = fold(c)
    for nom, f in ESCALERA.items():
        PRED[nom, c] = f(tr, va)
        filas.append(dict(modelo=nom, corte=c[:7], auc=roc_auc_score(va.target, PRED[nom, c])))
res = pd.DataFrame(filas).pivot(index="modelo", columns="corte", values="auc").loc[list(ESCALERA)]
res["promedio"] = res.mean(axis=1); res["mín"] = res.iloc[:, :6].min(axis=1); res["máx"] = res.iloc[:, :6].max(axis=1)
tabla(res.round(4), 12, "ROC-AUC por corte de validación (6 cortes walk-forward)")
''')
code(r'''
fig, ax = plt.subplots(figsize=(12, 5.2))
orden = list(ESCALERA)[::-1]
for i, nom in enumerate(orden):
    final = "FINAL" in nom
    c = VERDE if final else (GRIS if nom.startswith("M4 +") else AZUL)
    ax.plot([res.loc[nom, "mín"], res.loc[nom, "máx"]], [i, i], color=c, lw=2, alpha=.6)
    ax.scatter(res.loc[nom, res.columns[:6]], [i] * 6, color=c, s=14, alpha=.5)
    ax.scatter(res.loc[nom, "promedio"], i, color=c, s=110 if final else 70, zorder=3, edgecolor="white")
    ax.text(res.loc[nom, "máx"] + 0.0015, i, f"{res.loc[nom, 'promedio']:.4f}", va="center", fontsize=9,
            fontweight="bold" if final else "normal")
ax.set_yticks(range(len(orden)), orden)
ax.set(xlabel="ROC-AUC en validación (punto grande = promedio; puntos pequeños = cada corte)", xlim=(0.74, 0.80),
       title="Figura 9 — La escalera de modelos: de un groupby a LightGBM global")
plt.tight_layout(); plt.show()
''')
md("s5_2")

# ---------------------------------------------------------------- 6. Desbalance
md("s6")
code(r'''
CORTES_D = ["2018-08-01", "2019-03-01"]
CAT_IDX = [i for i, v in enumerate(VARIABLES) if v not in ("doy", "t")]

def estrategia(nombre, tr):
    rng = np.random.default_rng(SEED)
    pos, neg = tr[tr.target == 1], tr[tr.target == 0]
    if nombre == "sin rebalanceo":
        return lgbm(tr)
    if nombre == "class_weight = balanced":
        return lgb.LGBMClassifier(**PARAMS, class_weight="balanced").fit(tr[VARIABLES], tr.target)
    if nombre == "undersampling (positivos = 10 %)":
        d = pd.concat([pos, neg.iloc[rng.choice(len(neg), 9 * len(pos), replace=False)]])
    if nombre == "oversampling: bootstrap de positivos (30 %)":
        d = pd.concat([neg, pos.iloc[rng.integers(0, len(pos), int(0.3 * len(neg) / 0.7))]])
    if nombre == "SMOTENC (positivos = 10 %)":
        X = tr[VARIABLES].assign(barrio=tr.barrio.cat.codes)
        Xs, ys = SMOTENC(categorical_features=CAT_IDX, sampling_strategy=0.1 / 0.9, random_state=SEED).fit_resample(X, tr.target)
        Xs["barrio"] = pd.Categorical.from_codes(Xs.barrio.astype(int), tr.barrio.cat.categories)
        d = Xs.assign(target=ys.values)
    return lgbm(d)

ESTRATEGIAS = ["sin rebalanceo", "class_weight = balanced", "undersampling (positivos = 10 %)",
               "oversampling: bootstrap de positivos (30 %)", "SMOTENC (positivos = 10 %)"]
filas = []
for c in CORTES_D:
    tr, va = fold(c)
    for e in ESTRATEGIAS:
        s = estrategia(e, tr).predict_proba(va[VARIABLES])[:, 1]
        filas.append(dict(estrategia=e, corte=c[:7], **{"ROC-AUC": roc_auc_score(va.target, s), "PR-AUC": average_precision_score(va.target, s),
                                                       "prob. media predicha": s.mean(), "prevalencia real": va.target.mean()}))
des = pd.DataFrame(filas)
des_m = des.groupby("estrategia", sort=False)[["ROC-AUC", "PR-AUC", "prob. media predicha", "prevalencia real"]].mean()

fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
for j, (m, lim) in enumerate([("ROC-AUC", (0.775, 0.786)), ("PR-AUC", (0.055, 0.073))]):
    sns.barplot(data=des, y="estrategia", x=m, hue="corte", ax=ax[j], palette=[AZUL, NARANJA])
    ax[j].set(xlim=lim, ylabel="", title=f"({'ab'[j]}) {m}")
    ax[j].xaxis.set_major_locator(plt.MaxNLocator(4))
    if j: ax[j].set_yticklabels([])
    ax[j].legend(title="corte", fontsize=8, loc="lower right")
b = ax[2].barh(des_m.index[::-1], des_m["prob. media predicha"][::-1] * 100, color=AZUL)
ax[2].bar_label(b, fmt="%.1f %%", padding=3, fontsize=9)
ax[2].axvline(des_m["prevalencia real"].iloc[0] * 100, color=ROJO, ls="--"); ax[2].set_yticklabels([])
ax[2].text(des_m["prevalencia real"].iloc[0] * 100 + 0.6, 4.35, f"prevalencia real {des_m['prevalencia real'].iloc[0]:.1%}", color=ROJO, fontsize=9)
ax[2].set(xlim=(0, 45), xlabel="%", title="(c) Probabilidad media predicha")
fig.suptitle("Figura 10 — Estrategias de desbalance (LightGBM final, cortes 2018-08 y 2019-03)", y=1.02)
plt.tight_layout(); plt.show()
tabla(des_m.round(4), 13, "Estrategias de desbalance, promedio de los dos cortes")
''')
md("s6_1")
code(r'''
P18, P19 = PRED["M4 · LightGBM: barrio + calendario + t (FINAL)", "2018-08-01"], PRED["M4 · LightGBM: barrio + calendario + t (FINAL)", "2019-03-01"]
Y18, Y19 = fold("2018-08-01")[1].target.values, fold("2019-03-01")[1].target.values

def costo(y, s, umbral, k):
    tn, fp, fn, tp = confusion_matrix(y, s >= umbral, labels=[0, 1]).ravel()
    return fp + k * fn, (tn, fp, fn, tp)

grid = np.quantile(P18, np.linspace(0, 0.999, 400))
fig, ax = plt.subplots(figsize=(11, 4.8))
filas, UMBRAL = [], {}
for k, col in zip([10, 25, 50, 100], [GRIS, VERDE, NARANJA, VIOLETA]):
    sin_patrulla = costo(Y18, P18, 2, k)[0]
    cs = np.array([costo(Y18, P18, u, k)[0] for u in grid]) / sin_patrulla
    UMBRAL[k] = grid[cs.argmin()]
    ax.plot(grid, cs, color=col, lw=2.2 if k == 25 else 1.6, label=f"C_FN / C_FP = {k}")
    ax.scatter(UMBRAL[k], cs.min(), color=col, s=50, zorder=3)
    for nom, y, s in [("2018-08 (elige)", Y18, P18), ("2019-03 (aplica)", Y19, P19)]:
        c_, (tn, fp, fn, tp) = costo(y, s, UMBRAL[k], k)
        filas.append({"C_FN/C_FP": k, "umbral": round(UMBRAL[k], 4), "1/(1+k)": round(1 / (1 + k), 4), "corte": nom,
                      "% horas-barrio alertadas": round(100 * (tp + fp) / len(y), 1), "recall": round(tp / (tp + fn), 3),
                      "precision": round(tp / max(tp + fp, 1), 3), "ahorro vs no patrullar": f"{100 * (1 - c_ / costo(y, s, 2, k)[0]):.1f} %"})
ax.axhline(1, color="#333", lw=.8, ls=":")
ax.set(xscale="log", xlabel="umbral de probabilidad (escala log)", ylabel="costo / costo de no patrullar", ylim=(0.3, 2.2),
       title="Figura 11 — Costo esperado según el umbral (corte 2018-08); el punto marca el mínimo")
ax.legend(); plt.tight_layout(); plt.show()
tabla(pd.DataFrame(filas), 14, "Umbral elegido en 2018-08 y aplicado sin cambios en 2019-03")
''')
md("s6_2")

# ---------------------------------------------------------------- 7. Metricas
md("s7")
code(r'''
U = UMBRAL[25]
fig, ax = plt.subplots(1, 4, figsize=(20, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 0.8, 0.8]))
filas = []
for (nom, y, s), col, j in zip([("2018-08", Y18, P18), ("2019-03", Y19, P19)], [AZUL, NARANJA], [2, 3]):
    fpr, tpr, _ = roc_curve(y, s); pr, rc_, _ = precision_recall_curve(y, s)
    ax[0].plot(fpr, tpr, color=col, lw=2, label=f"{nom}: ROC-AUC {roc_auc_score(y, s):.3f}")
    ax[1].plot(rc_, pr, color=col, lw=2, label=f"{nom}: PR-AUC {average_precision_score(y, s):.3f}")
    cm = confusion_matrix(y, s >= U)
    sns.heatmap(cm, annot=np.array([[f"{v:,}" for v in r] for r in cm]), fmt="", cmap="Blues", cbar=False, ax=ax[j],
                xticklabels=["sin alerta", "alerta"], yticklabels=["sin accidente", "accidente"])
    ax[j].set(title=f"({'cd'[j - 2]}) Matriz de confusión {nom}", xlabel="predicción", ylabel="real" if j == 2 else "")
    tn, fp, fn, tp = cm.ravel()
    filas.append({"corte": nom, "umbral": round(U, 4), "precision": tp / (tp + fp), "recall": tp / (tp + fn), "F1": 2 * tp / (2 * tp + fp + fn),
                  "especificidad": tn / (tn + fp), "accuracy": (tp + tn) / cm.sum(), "accuracy de 'nunca hay accidente'": 1 - y.mean()})
ax[0].plot([0, 1], [0, 1], color=GRIS, ls="--", lw=1); ax[0].set(title="(a) Curva ROC", xlabel="tasa de falsos positivos", ylabel="recall")
ax[1].axhline(Y18.mean(), color=GRIS, ls="--", lw=1); ax[1].text(0.55, Y18.mean() + 0.004, "azar = prevalencia", color=GRIS, fontsize=9)
ax[1].set(title="(b) Curva precision-recall", xlabel="recall", ylabel="precision", ylim=(0, 0.2))
for a in ax[:2]: a.legend(fontsize=9)
fig.suptitle(f"Figura 12 — Métricas del modelo final en validación (umbral = {U:.4f}, escenario C_FN/C_FP = 25)", y=1.02)
plt.tight_layout(); plt.show()
tabla(pd.DataFrame(filas).set_index("corte").round(3), 15, "Métricas en el umbral elegido")
''')
md("s7_1")

# ---------------------------------------------------------------- 8. Resultado
md("s8")
code(r'''
tr_f = p[(p.TW < CORTE) & (p.target >= 0)]
te_f = p[(p.TW >= CORTE) & (p.TW <= "2019-12-30 23:00")].copy()
modelo_final = lgbm(tr_f)
te_f["prob"] = modelo_final.predict_proba(te_f[VARIABLES])[:, 1]
entrega = pd.DataFrame({"id": te_f.BARRIO + "|" + te_f.TW.dt.strftime("%Y-%m-%d %H:%M:%S"), "target": te_f.prob.values})
assert len(entrega) == 80256 and entrega.id.is_unique and entrega.target.between(0, 1).all()
assert set(entrega.id) == set(consulta.BARRIO + "|" + consulta.TW)            # exactamente las filas de la consulta del README
entrega.to_csv("mi_prediccion.csv", index=False)
print(f"Entrena con {len(tr_f):,} filas (hasta {tr_f.TW.max()}), predice {len(entrega):,} filas -> mi_prediccion.csv")

SUBIDO = "mi_prediccion_Savelezc.csv"                                      # archivo que se subió al leaderboard
if os.path.exists(SUBIDO):
    s = pd.read_csv(SUBIDO).merge(entrega, on="id", suffixes=("_subido", "_notebook"))
    print(f"Comparación con {SUBIDO}: {len(s):,} ids en común, diferencia máxima = {np.abs(s.target_subido - s.target_notebook).max():.2e}")
else:
    print(f"(Para comparar con lo subido al leaderboard, copie {SUBIDO} junto al notebook.)")
tabla(entrega.target.describe().to_frame("target").T.round(4), 16, "Resumen de las 80.256 probabilidades entregadas")
''')
md("s8_1")
code(r'''
LEADERBOARD_AUC = 0.7838                                                     # score oficial del leaderboard (check_status)
REF = {"tasa histórica barrio × hora (README)": 0.7589, "baseline del profesor (README)": 0.7739}
M4 = "M4 · LightGBM: barrio + calendario + t (FINAL)"
rng = np.random.default_rng(SEED)
d = fold("2018-08-01")[1].TW.dt.normalize().values
idx = pd.Series(np.arange(len(d))).groupby(d).indices
dias = np.array(list(idx))
boot = [roc_auc_score(Y18[j], P18[j]) for j in
        (np.concatenate([idx[x] for x in rng.choice(dias, len(dias), replace=True)]) for _ in range(1000))]
lo, hi = np.percentile(boot, [2.5, 97.5])

fig, ax = plt.subplots(1, 2, figsize=(17, 4.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
x = np.arange(6)
ax[0].scatter(x, res.loc[M4].iloc[:6], color=AZUL, s=60, zorder=3, label="validación (cada corte)")
ax[0].axhline(res.loc[M4, "promedio"], color=AZUL, lw=1.5, label=f"promedio validación = {res.loc[M4, 'promedio']:.4f}")
ax[0].axhspan(lo, hi, color=AZUL, alpha=.12, label=f"rango 95 % por azar de un periodo de 5 meses [{lo:.3f}, {hi:.3f}]")
ax[0].scatter(6.2, LEADERBOARD_AUC, marker="*", s=380, color=VERDE, zorder=4, edgecolor="#333", label=f"LEADERBOARD ago-dic 2019 = {LEADERBOARD_AUC}")
for (nom, v), c in zip(REF.items(), [GRIS, ROJO]):
    ax[0].axhline(v, color=c, ls="--", lw=1.2, label=f"{nom} = {v}")
ax[0].set_xticks(list(x) + [6.2], [c[:7] for c in CORTES] + ["leaderboard\n2019"])
ax[0].set(ylabel="ROC-AUC", ylim=(0.755, 0.80), title="(a) Validación vs. resultado oficial")
ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
imp = pd.Series(modelo_final.booster_.feature_importance("gain"), index=VARIABLES)
imp = (100 * imp / imp.sum()).sort_values()
b = ax[1].barh(imp.index, imp.values, color=AZUL); ax[1].bar_label(b, fmt="%.1f%%", padding=2, fontsize=8)
ax[1].set(xlabel="% de la ganancia total de los splits", title="(b) Importancia de las variables en el modelo final")
fig.suptitle("Figura 13 — RESULTADO OFICIAL", y=1.02)
plt.tight_layout(); plt.show()
tabla(pd.DataFrame({"ROC-AUC": [LEADERBOARD_AUC, res.loc[M4, "promedio"], res.loc[M4, "2018-08"], lo, hi, *REF.values()]},
                   index=["LEADERBOARD (ago-dic 2019)", "validación: promedio 6 cortes", "validación: corte 2018-08 (misma temporada)",
                          "rango 95 % por azar: límite inferior", "rango 95 % por azar: límite superior", *REF]).round(4),
      17, "Resultado oficial frente a la validación y las referencias")
''')

# ---------------------------------------------------------------- 9. Caso de uso
md("s9")
code(r'''
te_f["tipo"] = np.where((te_f.festivo == 1) | (te_f.dow >= 5), "fin de semana o festivo", "laboral")
fig, ax = plt.subplots(1, 2, figsize=(19, 7), sharey=True)
orden_b = te_f.groupby("BARRIO").prob.mean().sort_values(ascending=False).index
for j, t in enumerate(["laboral", "fin de semana o festivo"]):
    hm = te_f[te_f.tipo == t].pivot_table(index="BARRIO", columns="h", values="prob", aggfunc="mean").loc[orden_b] * 100
    sns.heatmap(hm, cmap="YlOrRd", vmin=0, vmax=12, ax=ax[j], cbar=j == 1, cbar_kws=dict(label="probabilidad de ≥ 1 accidente (%)"))
    ax[j].set(title=f"({'ab'[j]}) Día {t}", xlabel="hora", ylabel="" if j else "barrio")
fig.suptitle("Figura 14 — Mapa de calor del riesgo predicho para ago-dic 2019 (promedio por barrio y hora)", y=1.0)
plt.tight_layout(); plt.show()
''')
md("s9_0")
code(r'''
va19 = fold("2019-03-01")[1].assign(prob=P19)
va19["rk"] = va19.groupby(va19.TW.dt.normalize()).prob.rank(ascending=False, method="first")
filas = []
for n in [1, 2, 3, 5, 10]:
    cub = va19.rk <= 4 * n
    filas.append({"patrullas × 4 h/día": n, "horas-barrio cubiertas por día": 4 * n, "% de las 528 diarias": round(100 * 4 * n / 528, 1),
                  "% de accidentes cubiertos": round(100 * va19.target[cub].sum() / va19.target.sum(), 1),
                  "lift frente a patrullar al azar": round(va19.target[cub].mean() / va19.target.mean(), 1)})
tabla(pd.DataFrame(filas).set_index("patrullas × 4 h/día"), 18, "Cobertura con N patrullas de 4 horas (validación mar-jul 2019)")
dia = te_f[te_f.TW.dt.normalize() == "2019-08-05"].nlargest(12, "prob")[["TW", "BARRIO", "prob"]]
dia["prob"] = (dia.prob * 100).round(2)
tabla(dia.rename(columns={"prob": "prob. (%)"}).reset_index(drop=True), 19, "Ejemplo de turno: las 12 horas-barrio de mayor riesgo el lunes 2019-08-05")
''')
md("s9_1")
md("resumen")

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t) for k, t in C]
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
nbf.write(nb, "Reto2_Accidentalidad_Poblado.ipynb")
print(len(nb.cells), "celdas")
