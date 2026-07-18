"""
Dashboard COVID-19 — Datos sintéticos, métricas y análisis gráfico.
Ejecutar con:  streamlit run main_app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ----------------------------------------------------------------------
# 1. CONFIGURACIÓN GENERAL
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard COVID-19 sintético",
    page_icon="🧬",
    layout="wide",
)

PALETAS = ["Plotly", "Viridis", "Turbo", "IceFire", "Sunset", "Bold"]


# ----------------------------------------------------------------------
# 2. SIMULACIÓN DE DATOS (10.000 registros × 8 columnas)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Simulando datos...")
def simular_datos(n: int, semilla: int) -> pd.DataFrame:
    """Genera un dataset sintético de casos COVID con 8 columnas y tipos mixtos."""
    rng = np.random.default_rng(semilla)

    # --- Columna 1: fecha (datetime) ---
    inicio = pd.Timestamp("2020-03-01")
    dias = rng.integers(0, 1000, n)
    # Estacionalidad: más casos en olas periódicas
    peso_ola = 1 + 0.6 * np.sin(dias / 60.0)
    dias = rng.choice(dias, size=n, replace=True, p=peso_ola / peso_ola.sum())
    fecha_reporte = inicio + pd.to_timedelta(dias, unit="D")

    # --- Columna 2: región (categórica nominal) ---
    regiones = ["Andina", "Caribe", "Pacífica", "Orinoquía", "Amazonía"]
    region = rng.choice(regiones, n, p=[0.45, 0.22, 0.18, 0.09, 0.06])

    # --- Columna 3: sexo (categórica binaria) ---
    sexo = rng.choice(["Femenino", "Masculino"], n, p=[0.51, 0.49])

    # --- Columna 4: edad (entero, distribución gamma truncada) ---
    edad = np.clip(rng.gamma(shape=6.0, scale=7.0, size=n), 0, 99).astype(int)

    # --- Columna 5: vacunado (booleano, depende de la edad) ---
    p_vac = np.clip(0.25 + edad / 140.0, 0.05, 0.95)
    vacunado = rng.random(n) < p_vac

    # --- Columna 6: carga viral, log10 copias/mL (float) ---
    carga_viral = np.round(
        rng.normal(5.4, 1.1, n) - 0.8 * vacunado + 0.012 * (edad - 40), 2
    )
    carga_viral = np.clip(carga_viral, 1.0, 9.5)

    # --- Columna 7: días con síntomas (entero, Poisson dependiente) ---
    lam = np.clip(3 + 0.09 * edad + 1.2 * (carga_viral - 5) - 2.0 * vacunado, 0.5, None)
    dias_sintomas = rng.poisson(lam)

    # --- Columna 8: estado clínico (categórica ordinal, dependiente) ---
    riesgo = 0.02 * edad + 0.35 * (carga_viral - 5) - 1.1 * vacunado + rng.normal(0, 0.8, n)
    estado_clinico = np.select(
        [riesgo < 0.6, riesgo < 1.5, riesgo < 2.4],
        ["Leve", "Moderado", "Grave"],
        default="Crítico",
    )

    df = pd.DataFrame(
        {
            "fecha_reporte": fecha_reporte,
            "region": pd.Categorical(region),
            "sexo": pd.Categorical(sexo),
            "edad": edad,
            "vacunado": vacunado,
            "carga_viral": carga_viral,
            "dias_sintomas": dias_sintomas,
            "estado_clinico": pd.Categorical(
                estado_clinico, categories=["Leve", "Moderado", "Grave", "Crítico"], ordered=True
            ),
        }
    )
    return df.sort_values("fecha_reporte").reset_index(drop=True)


# ----------------------------------------------------------------------
# 3. BARRA LATERAL — CONTROLES DEL USUARIO
# ----------------------------------------------------------------------
st.sidebar.title("⚙️ Controles")

with st.sidebar.expander("Simulación", expanded=True):
    n_registros = st.slider("Número de registros", 1_000, 50_000, 10_000, step=1_000)
    semilla = st.number_input("Semilla aleatoria", 0, 9_999, 42)
    if st.button("🔄 Volver a simular"):
        st.cache_data.clear()

df = simular_datos(n_registros, semilla)

NUMERICAS = ["edad", "carga_viral", "dias_sintomas"]
CATEGORICAS = ["region", "sexo", "estado_clinico", "vacunado"]

with st.sidebar.expander("Filtros", expanded=True):
    rango_fechas = st.date_input(
        "Rango de fechas",
        value=(df["fecha_reporte"].min().date(), df["fecha_reporte"].max().date()),
        min_value=df["fecha_reporte"].min().date(),
        max_value=df["fecha_reporte"].max().date(),
    )
    regiones_sel = st.multiselect(
        "Regiones", list(df["region"].cat.categories), default=list(df["region"].cat.categories)
    )
    edad_min, edad_max = st.slider("Rango de edad", 0, 99, (0, 99))
    solo_vacunados = st.selectbox("Estado de vacunación", ["Todos", "Vacunados", "No vacunados"])

with st.sidebar.expander("Umbrales de alerta", expanded=True):
    umbral_var = st.selectbox("Variable del umbral", NUMERICAS, index=1)
    lim_inf = float(df[umbral_var].min())
    lim_sup = float(df[umbral_var].max())
    umbral = st.slider(
        f"Umbral sobre {umbral_var}",
        lim_inf,
        lim_sup,
        float(np.percentile(df[umbral_var], 75)),
    )

with st.sidebar.expander("Apariencia", expanded=False):
    paleta = st.selectbox("Paleta de colores", PALETAS)
    plantilla = st.selectbox(
        "Tema del gráfico", ["plotly_white", "plotly_dark", "simple_white", "ggplot2", "seaborn"]
    )
    bins = st.slider("Bins del histograma", 5, 120, 40)
    opacidad = st.slider("Opacidad de las marcas", 0.2, 1.0, 0.75)

# --- Aplicar filtros ---
mask = (
    df["region"].isin(regiones_sel)
    & df["edad"].between(edad_min, edad_max)
    & df["fecha_reporte"].between(
        pd.Timestamp(rango_fechas[0]), pd.Timestamp(rango_fechas[-1]) + pd.Timedelta(days=1)
    )
)
if solo_vacunados == "Vacunados":
    mask &= df["vacunado"]
elif solo_vacunados == "No vacunados":
    mask &= ~df["vacunado"]

dff = df[mask].copy()
dff["supera_umbral"] = dff[umbral_var] > umbral

if dff.empty:
    st.warning("Ningún registro cumple los filtros. Ajusta los controles de la izquierda.")
    st.stop()


def color_kwargs():
    """Devuelve la secuencia de colores discreta según la paleta elegida."""
    for modulo in (px.colors.qualitative, px.colors.sequential, px.colors.diverging):
        secuencia = getattr(modulo, paleta, None)
        if secuencia:
            return {"color_discrete_sequence": list(secuencia)}
    return {"color_discrete_sequence": px.colors.qualitative.Plotly}


# ----------------------------------------------------------------------
# 4. ENCABEZADO Y KPIs
# ----------------------------------------------------------------------
st.title("🧬 Dashboard COVID-19 — datos sintéticos")
st.caption(
    f"{len(dff):,} de {len(df):,} registros tras aplicar filtros · "
    f"umbral: {umbral_var} > {umbral:.2f}"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Casos", f"{len(dff):,}")
k2.metric("Edad media", f"{dff['edad'].mean():.1f} años")
k3.metric("Carga viral media", f"{dff['carga_viral'].mean():.2f} log₁₀")
k4.metric("Vacunados", f"{dff['vacunado'].mean() * 100:.1f} %")
k5.metric("Sobre umbral", f"{dff['supera_umbral'].mean() * 100:.1f} %")

tab_datos, tab_cuant, tab_cual, tab_graf, tab_rel = st.tabs(
    ["📄 Datos", "📐 Cuantitativo", "🏷️ Cualitativo", "📊 Gráficos", "🔗 Relaciones"]
)

# ----------------------------------------------------------------------
# 5. PESTAÑA — DATOS
# ----------------------------------------------------------------------
with tab_datos:
    st.subheader("Muestra del dataset")
    st.dataframe(dff.head(200), use_container_width=True)

    st.subheader("Diccionario de variables")
    dicc = pd.DataFrame(
        {
            "variable": dff.columns[:8],
            "tipo_pandas": [str(t) for t in dff.dtypes[:8]],
            "escala": [
                "Temporal", "Nominal", "Nominal", "Razón",
                "Binaria", "Razón", "Razón", "Ordinal",
            ],
            "no_nulos": [dff[c].notna().sum() for c in dff.columns[:8]],
            "únicos": [dff[c].nunique() for c in dff.columns[:8]],
        }
    )
    st.dataframe(dicc, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Descargar CSV filtrado",
        dff.to_csv(index=False).encode("utf-8"),
        file_name="covid_sintetico.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------------
# 6. PESTAÑA — ESTADÍSTICA CUANTITATIVA
# ----------------------------------------------------------------------
with tab_cuant:
    st.subheader("Esquema de métricas cuantitativas")
    filas = []
    for c in NUMERICAS:
        s = dff[c].astype(float)
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        atipicos = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        filas.append(
            {
                "variable": c,
                "n": s.size,
                "media": s.mean(),
                "mediana": s.median(),
                "desv_est": s.std(),
                "error_est": s.sem(),
                "CV %": s.std() / s.mean() * 100 if s.mean() else np.nan,
                "mín": s.min(),
                "Q1": q1,
                "Q3": q3,
                "máx": s.max(),
                "IQR": iqr,
                "asimetría": stats.skew(s),
                "curtosis": stats.kurtosis(s),
                "atípicos": atipicos,
                "IC95 inf": s.mean() - 1.96 * s.sem(),
                "IC95 sup": s.mean() + 1.96 * s.sem(),
            }
        )
    st.dataframe(
        pd.DataFrame(filas).set_index("variable").round(3), use_container_width=True
    )

    st.markdown("**Interpretación rápida:** un CV mayor a 30 % indica alta dispersión; "
                "una asimetría fuera de ±0.5 sugiere una distribución sesgada.")

    st.divider()
    st.subheader("Prueba de hipótesis")
    c1, c2 = st.columns(2)
    var_t = c1.selectbox("Variable numérica", NUMERICAS, key="ttest_var")
    grupo_t = c2.selectbox("Variable de agrupación (2 grupos)", ["vacunado", "sexo"])

    grupos = dff.groupby(grupo_t, observed=True)[var_t].apply(list)
    if len(grupos) == 2:
        a, b = grupos.iloc[0], grupos.iloc[1]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        c1.metric("t de Welch", f"{t:.3f}")
        c2.metric("p-valor", f"{p:.2e}")
        st.write(
            f"d de Cohen = **{d:.3f}** · "
            + ("Diferencia estadísticamente significativa (α = 0.05)."
               if p < 0.05 else "Sin evidencia de diferencia (α = 0.05).")
        )

# ----------------------------------------------------------------------
# 7. PESTAÑA — ESTADÍSTICA CUALITATIVA
# ----------------------------------------------------------------------
with tab_cual:
    st.subheader("Tabla de frecuencias")
    var_cual = st.selectbox("Variable categórica", CATEGORICAS)

    freq = dff[var_cual].value_counts().rename_axis("categoría").reset_index(name="frecuencia")
    freq["% relativo"] = (freq["frecuencia"] / freq["frecuencia"].sum() * 100).round(2)
    freq["% acumulado"] = freq["% relativo"].cumsum().round(2)
    st.dataframe(freq, use_container_width=True, hide_index=True)

    p = freq["frecuencia"] / freq["frecuencia"].sum()
    entropia = -(p * np.log2(p)).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Moda", str(freq.iloc[0]["categoría"]))
    c2.metric("Categorías", freq.shape[0])
    c3.metric("Entropía de Shannon", f"{entropia:.3f} bits")

    st.divider()
    st.subheader("Tabla de contingencia y χ²")
    c1, c2 = st.columns(2)
    v1 = c1.selectbox("Variable fila", CATEGORICAS, index=0, key="chi1")
    v2 = c2.selectbox("Variable columna", CATEGORICAS, index=2, key="chi2")

    tabla = pd.crosstab(dff[v1], dff[v2])
    st.dataframe(tabla, use_container_width=True)

    if tabla.shape[0] > 1 and tabla.shape[1] > 1:
        chi2, pval, gl, _ = stats.chi2_contingency(tabla)
        n = tabla.values.sum()
        cramer = np.sqrt(chi2 / (n * (min(tabla.shape) - 1)))
        a, b, c = st.columns(3)
        a.metric("χ²", f"{chi2:.2f}")
        b.metric("p-valor", f"{pval:.2e}")
        c.metric("V de Cramér", f"{cramer:.3f}")

        fig = px.imshow(
            tabla, text_auto=True, aspect="auto", template=plantilla,
            color_continuous_scale=paleta if paleta not in ("Plotly", "Bold") else "Blues",
            title=f"Mapa de calor: {v1} × {v2}",
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# 8. PESTAÑA — ANÁLISIS GRÁFICO
# ----------------------------------------------------------------------
with tab_graf:
    st.subheader("Distribución univariada")
    c1, c2 = st.columns([1, 2])
    var_dist = c1.selectbox("Variable a pintar", NUMERICAS, key="dist")
    color_por = c1.selectbox("Colorear por", ["(ninguno)"] + CATEGORICAS, key="dist_color")
    tipo = c1.radio("Tipo de gráfico", ["Histograma", "Caja", "Violín", "ECDF"])
    marcar_umbral = c1.checkbox("Marcar línea de umbral", value=True)

    col = None if color_por == "(ninguno)" else color_por

    if tipo == "Histograma":
        fig = px.histogram(dff, x=var_dist, color=col, nbins=bins, marginal="box",
                           barmode="overlay", opacity=opacidad, template=plantilla,
                           **color_kwargs())
    elif tipo == "Caja":
        fig = px.box(dff, y=var_dist, color=col, points="outliers",
                     template=plantilla, **color_kwargs())
    elif tipo == "Violín":
        fig = px.violin(dff, y=var_dist, color=col, box=True, points=False,
                        template=plantilla, **color_kwargs())
    else:
        fig = px.ecdf(dff, x=var_dist, color=col, template=plantilla, **color_kwargs())

    if marcar_umbral and var_dist == umbral_var:
        etiqueta = f"umbral = {umbral:.2f}"
        if tipo in ("Histograma", "ECDF"):
            fig.add_vline(x=umbral, line_dash="dash", line_color="crimson",
                          annotation_text=etiqueta)
        else:
            fig.add_hline(y=umbral, line_dash="dash", line_color="crimson",
                          annotation_text=etiqueta)

    c2.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Composición por categoría")
    c1, c2 = st.columns(2)
    var_bar = c1.selectbox("Variable de barras", CATEGORICAS, key="bar")
    agrupar = c2.selectbox("Desagregar por", ["(ninguno)"] + CATEGORICAS, key="bar_grp")
    modo = c1.radio("Modo de barras", ["group", "stack", "relative"], horizontal=True)

    conteo = (
        dff.groupby([var_bar] + ([] if agrupar == "(ninguno)" else [agrupar]), observed=True)
        .size()
        .reset_index(name="casos")
    )
    fig_bar = px.bar(
        conteo, x=var_bar, y="casos",
        color=None if agrupar == "(ninguno)" else agrupar,
        barmode=modo, text_auto=True, template=plantilla,
        opacity=opacidad, **color_kwargs(),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.subheader("Serie temporal")
    c1, c2, c3 = st.columns(3)
    frec = c1.selectbox("Agregación", {"Diaria": "D", "Semanal": "W", "Mensual": "MS"}.keys())
    metrica_t = c2.selectbox("Métrica", ["Casos", "Media de la variable"], key="serie_m")
    var_t2 = c3.selectbox("Variable (si aplica)", NUMERICAS, key="serie_v")

    codigo = {"Diaria": "D", "Semanal": "W", "Mensual": "MS"}[frec]
    serie = dff.set_index("fecha_reporte")
    if metrica_t == "Casos":
        serie = serie.resample(codigo).size().reset_index(name="valor")
    else:
        serie = serie[var_t2].resample(codigo).mean().reset_index(name="valor")

    fig_t = px.line(serie, x="fecha_reporte", y="valor", markers=False,
                    template=plantilla, **color_kwargs())
    fig_t.update_traces(line_width=2)
    st.plotly_chart(fig_t, use_container_width=True)

# ----------------------------------------------------------------------
# 9. PESTAÑA — RELACIONES ENTRE VARIABLES
# ----------------------------------------------------------------------
with tab_rel:
    st.subheader("Dispersión bivariada")
    c1, c2, c3, c4 = st.columns(4)
    x = c1.selectbox("Eje X", NUMERICAS, index=0)
    y = c2.selectbox("Eje Y", NUMERICAS, index=1)
    col_sc = c3.selectbox("Color", ["(ninguno)", "supera_umbral"] + CATEGORICAS)
    tend = c4.selectbox("Línea de tendencia", ["ninguna", "ols", "lowess"])

    muestra = dff.sample(min(3000, len(dff)), random_state=1)  # rendimiento
    fig_sc = px.scatter(
        muestra, x=x, y=y,
        color=None if col_sc == "(ninguno)" else col_sc,
        opacity=opacidad, template=plantilla,
        trendline=None if tend == "ninguna" else tend,
        trendline_color_override="crimson",
        **color_kwargs(),
    )
    if x == umbral_var:
        fig_sc.add_vline(x=umbral, line_dash="dash", line_color="crimson")
    if y == umbral_var:
        fig_sc.add_hline(y=umbral, line_dash="dash", line_color="crimson")
    st.plotly_chart(fig_sc, use_container_width=True)

    r, p_r = stats.pearsonr(dff[x], dff[y])
    rho, p_s = stats.spearmanr(dff[x], dff[y])
    a, b = st.columns(2)
    a.metric("Pearson r", f"{r:.3f}", help=f"p = {p_r:.2e}")
    b.metric("Spearman ρ", f"{rho:.3f}", help=f"p = {p_s:.2e}")

    st.divider()
    st.subheader("Matriz de correlación")
    metodo = st.radio("Método", ["pearson", "spearman", "kendall"], horizontal=True)
    corr = dff[NUMERICAS].corr(method=metodo).round(3)
    fig_c = px.imshow(corr, text_auto=True, zmin=-1, zmax=1,
                      color_continuous_scale="RdBu_r", template=plantilla)
    st.plotly_chart(fig_c, use_container_width=True)
