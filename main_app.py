"""
Dashboard COVID-19 — EAFIT 2026 · Ciencia de Datos
Clave de acceso: 4650
Ejecutar con:  streamlit run main_app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ======================================================================
# 1. CONFIGURACIÓN GENERAL
# ======================================================================
st.set_page_config(
    page_title="COVID-19 · EAFIT 2026",
    page_icon="🧬",
    layout="wide",
)

PALETAS = ["Plotly", "Viridis", "Turbo", "IceFire", "Sunset", "Bold"]
CLAVE_CORRECTA = "4650"

# ======================================================================
# 2. PANTALLA DE ACCESO (clave 4650)
# ======================================================================
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.markdown(
        """
        <div style='max-width:420px; margin:10vh auto; padding:2.5rem 2rem;
                    background:#0e1117; border:1px solid #31333f; border-radius:12px;
                    text-align:center;'>
            <h1 style='color:#FFFFFF; font-size:2rem; margin-bottom:0.25rem;'>🧬 COVID-19</h1>
            <p style='color:#888; font-size:0.85rem; margin-bottom:2rem;'>
                EAFIT 2026 · Ciencia de Datos<br>
                Profesor Jorge Padilla · Julio 2026
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    clave_input = st.text_input(
        "🔐 Ingresa la clave de acceso", type="password", max_chars=10,
        placeholder="••••"
    )
    if st.button("Ingresar"):
        if clave_input == CLAVE_CORRECTA:
            st.session_state.autenticado = True
            st.rerun()
        else:
            st.error("Clave incorrecta. Inténtalo de nuevo.")
    st.stop()

# ======================================================================
# 3. SIMULACIÓN DE DATOS (10 000 registros × 8 columnas)
# ======================================================================
@st.cache_data(show_spinner="⏳ Simulando datos COVID…")
def simular_datos(n: int, semilla: int) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    # Fechas con 3 olas estacionales
    inicio = pd.Timestamp("2020-03-01")
    t = np.linspace(0, 6 * np.pi, n)
    peso_ola = np.abs(np.sin(t)) + 0.15
    peso_ola /= peso_ola.sum()
    dias_base = np.arange(n) * (1000 / n)
    dias = rng.choice(dias_base.astype(int), size=n, replace=True, p=peso_ola)
    fecha_reporte = inicio + pd.to_timedelta(dias, unit="D")

    regiones = ["Andina", "Caribe", "Pacífica", "Orinoquía", "Amazonía"]
    region   = rng.choice(regiones, n, p=[0.45, 0.22, 0.18, 0.09, 0.06])
    sexo     = rng.choice(["Femenino", "Masculino"], n, p=[0.51, 0.49])
    edad     = np.clip(rng.gamma(shape=6.0, scale=7.0, size=n), 0, 99).astype(int)

    p_vac    = np.clip(0.25 + edad / 140.0, 0.05, 0.95)
    vacunado = rng.random(n) < p_vac

    carga_viral = np.round(
        rng.normal(5.4, 1.1, n) - 0.8 * vacunado + 0.012 * (edad - 40), 2
    )
    carga_viral = np.clip(carga_viral, 1.0, 9.5)

    lam = np.clip(3 + 0.09 * edad + 1.2 * (carga_viral - 5) - 2.0 * vacunado, 0.5, None)
    dias_sintomas = rng.poisson(lam)

    riesgo = 0.02 * edad + 0.35 * (carga_viral - 5) - 1.1 * vacunado + rng.normal(0, 0.8, n)
    estado_clinico = np.select(
        [riesgo < 0.6, riesgo < 1.5, riesgo < 2.4],
        ["Leve", "Moderado", "Grave"],
        default="Crítico",
    )

    df = pd.DataFrame({
        "fecha_reporte": fecha_reporte,
        "region":        pd.Categorical(region),
        "sexo":          pd.Categorical(sexo),
        "edad":          edad,
        "vacunado":      vacunado,
        "carga_viral":   carga_viral,
        "dias_sintomas": dias_sintomas,
        "estado_clinico": pd.Categorical(
            estado_clinico,
            categories=["Leve", "Moderado", "Grave", "Crítico"],
            ordered=True,
        ),
    })
    return df.sort_values("fecha_reporte").reset_index(drop=True)

# ======================================================================
# 4. BARRA LATERAL — IDENTIDAD + CONTROLES
# ======================================================================
st.sidebar.markdown(
    """
    <div style='text-align:center; padding: 0.8rem 0 1rem 0;'>
        <div style='font-size:2rem;'>🧬</div>
        <div style='font-weight:800; font-size:1.15rem; color:#FFFFFF; letter-spacing:0.04em;'>
            EAFIT 2026
        </div>
        <div style='color:#aaa; font-size:0.82rem; margin-top:0.1rem;'>Ciencia de Datos</div>
        <hr style='border:none; border-top:1px solid #333; margin:0.7rem 0;'>
        <div style='color:#ccc; font-size:0.78rem; line-height:1.7;'>
            👨‍🏫 Profesor Jorge Padilla<br>
            📅 Julio de 2026
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("🚪 Cerrar sesión"):
    st.session_state.autenticado = False
    st.rerun()

st.sidebar.divider()

with st.sidebar.expander("🎲 Simulación", expanded=True):
    n_registros = st.slider("Número de registros", 1_000, 50_000, 10_000, step=1_000)
    semilla     = st.number_input("Semilla aleatoria", 0, 9_999, 42)
    if st.button("🔄 Volver a simular"):
        st.cache_data.clear()

df = simular_datos(n_registros, semilla)
NUMERICAS   = ["edad", "carga_viral", "dias_sintomas"]
CATEGORICAS = ["region", "sexo", "estado_clinico", "vacunado"]

with st.sidebar.expander("🔎 Filtros", expanded=True):
    rango_fechas = st.date_input(
        "Rango de fechas",
        value=(df["fecha_reporte"].min().date(), df["fecha_reporte"].max().date()),
        min_value=df["fecha_reporte"].min().date(),
        max_value=df["fecha_reporte"].max().date(),
    )
    regiones_sel = st.multiselect(
        "Regiones", list(df["region"].cat.categories),
        default=list(df["region"].cat.categories),
    )
    edad_min, edad_max = st.slider("Rango de edad", 0, 99, (0, 99))
    solo_vac = st.selectbox("Vacunación", ["Todos", "Vacunados", "No vacunados"])

with st.sidebar.expander("🚨 Umbral de alerta", expanded=True):
    umbral_var = st.selectbox("Variable", NUMERICAS, index=1)
    lim_inf    = float(df[umbral_var].min())
    lim_sup    = float(df[umbral_var].max())
    umbral     = st.slider(
        f"Umbral en {umbral_var}", lim_inf, lim_sup,
        float(np.percentile(df[umbral_var], 75)),
    )

with st.sidebar.expander("🎨 Apariencia", expanded=False):
    paleta    = st.selectbox("Paleta", PALETAS)
    plantilla = st.selectbox(
        "Tema", ["plotly_white", "plotly_dark", "simple_white", "ggplot2", "seaborn"]
    )
    bins     = st.slider("Bins del histograma", 5, 120, 40)
    opacidad = st.slider("Opacidad", 0.2, 1.0, 0.75)

# --- Aplicar filtros ---
mask = (
    df["region"].isin(regiones_sel)
    & df["edad"].between(edad_min, edad_max)
    & df["fecha_reporte"].between(
        pd.Timestamp(rango_fechas[0]),
        pd.Timestamp(rango_fechas[-1]) + pd.Timedelta(days=1),
    )
)
if solo_vac == "Vacunados":
    mask &= df["vacunado"]
elif solo_vac == "No vacunados":
    mask &= ~df["vacunado"]

dff = df[mask].copy()
dff["supera_umbral"] = dff[umbral_var] > umbral

if dff.empty:
    st.warning("⚠️ Ningún registro cumple los filtros. Ajusta los controles.")
    st.stop()


def color_kwargs():
    for mod in (px.colors.qualitative, px.colors.sequential, px.colors.diverging):
        seq = getattr(mod, paleta, None)
        if seq:
            return {"color_discrete_sequence": list(seq)}
    return {"color_discrete_sequence": px.colors.qualitative.Plotly}


# ======================================================================
# 5. ENCABEZADO Y KPIs
# ======================================================================
st.title("🧬 Dashboard COVID-19 — Datos Sintéticos")
st.caption(
    f"**EAFIT 2026 · Ciencia de Datos · Profesor Jorge Padilla**  |  "
    f"{len(dff):,} de {len(df):,} registros filtrados  ·  umbral: {umbral_var} > {umbral:.2f}"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📋 Casos", f"{len(dff):,}")
k2.metric("🎂 Edad media", f"{dff['edad'].mean():.1f} a")
k3.metric("🦠 Carga viral media", f"{dff['carga_viral'].mean():.2f} log₁₀")
k4.metric("💉 Vacunados", f"{dff['vacunado'].mean() * 100:.1f} %")
k5.metric("🚨 Sobre umbral", f"{dff['supera_umbral'].mean() * 100:.1f} %")

tab_datos, tab_serie, tab_cuant, tab_cual, tab_graf, tab_rel = st.tabs([
    "📄 Datos", "📈 Serie temporal", "📐 Cuantitativo",
    "🏷️ Cualitativo", "📊 Gráficos", "🔗 Relaciones",
])

# ======================================================================
# 6. PESTAÑA — DATOS
# ======================================================================
with tab_datos:
    st.subheader("Muestra del dataset")
    st.dataframe(dff.head(300), use_container_width=True)

    st.subheader("Diccionario de variables")
    dicc = pd.DataFrame({
        "variable":    list(dff.columns[:8]),
        "tipo_pandas": [str(t) for t in dff.dtypes[:8]],
        "escala":      ["Temporal","Nominal","Nominal","Razón","Binaria","Razón","Razón","Ordinal"],
        "no_nulos":    [dff[c].notna().sum() for c in dff.columns[:8]],
        "únicos":      [dff[c].nunique()      for c in dff.columns[:8]],
    })
    st.dataframe(dicc, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Descargar CSV filtrado",
        dff.to_csv(index=False).encode("utf-8"),
        file_name="covid_sintetico.csv", mime="text/csv",
    )

# ======================================================================
# 7. PESTAÑA — SERIE TEMPORAL (nueva, completa e interactiva)
# ======================================================================
with tab_serie:
    st.subheader("📈 Serie temporal — Evolución de la epidemia")

    # --- Controles de la serie ---
    col_a, col_b, col_c, col_d = st.columns(4)
    frec_map  = {"Diaria": "D", "Semanal": "W", "Mensual": "MS"}
    frec_lbl  = col_a.selectbox("Agregación temporal", list(frec_map.keys()), index=1, key="st_frec")
    metrica_s = col_b.selectbox("Métrica", ["Casos nuevos", "Media carga viral",
                                             "Media días síntomas", "% vacunados"], key="st_met")
    desglose  = col_c.selectbox("Desglosar por", ["(ninguno)", "region", "sexo",
                                                    "estado_clinico"], key="st_des")
    suavizar  = col_d.slider("Suavizado (media móvil, períodos)", 1, 12, 1, key="st_suav")

    codigo = frec_map[frec_lbl]
    idx    = dff.set_index("fecha_reporte")

    if desglose == "(ninguno)":
        if metrica_s == "Casos nuevos":
            serie_df = idx.resample(codigo).size().reset_index(name="valor")
        elif metrica_s == "Media carga viral":
            serie_df = idx["carga_viral"].resample(codigo).mean().reset_index(name="valor")
        elif metrica_s == "Media días síntomas":
            serie_df = idx["dias_sintomas"].resample(codigo).mean().reset_index(name="valor")
        else:
            serie_df = idx["vacunado"].resample(codigo).mean().mul(100).reset_index(name="valor")

        if suavizar > 1:
            serie_df["valor"] = serie_df["valor"].rolling(suavizar, min_periods=1).mean()

        fig_s = px.line(
            serie_df, x="fecha_reporte", y="valor",
            labels={"fecha_reporte": "Fecha", "valor": metrica_s},
            title=f"{metrica_s} — {frec_lbl}",
            template=plantilla,
        )
        fig_s.update_traces(line_width=2.5)

    else:
        filas = []
        for grp, sub in idx.groupby(desglose, observed=True):
            if metrica_s == "Casos nuevos":
                tmp = sub.resample(codigo).size().reset_index(name="valor")
            elif metrica_s == "Media carga viral":
                tmp = sub["carga_viral"].resample(codigo).mean().reset_index(name="valor")
            elif metrica_s == "Media días síntomas":
                tmp = sub["dias_sintomas"].resample(codigo).mean().reset_index(name="valor")
            else:
                tmp = sub["vacunado"].resample(codigo).mean().mul(100).reset_index(name="valor")
            tmp[desglose] = str(grp)
            if suavizar > 1:
                tmp["valor"] = tmp["valor"].rolling(suavizar, min_periods=1).mean()
            filas.append(tmp)

        serie_df = pd.concat(filas, ignore_index=True)
        fig_s = px.line(
            serie_df, x="fecha_reporte", y="valor", color=desglose,
            labels={"fecha_reporte": "Fecha", "valor": metrica_s},
            title=f"{metrica_s} por {desglose} — {frec_lbl}",
            template=plantilla, **color_kwargs(),
        )
        fig_s.update_traces(line_width=2)

    # Área rellena bajo la curva si no hay desglose
    if desglose == "(ninguno)":
        fig_s.update_traces(fill="tozeroy", fillcolor="rgba(99,110,250,0.15)")

    # Línea de umbral si la métrica es carga viral
    if metrica_s == "Media carga viral":
        fig_s.add_hline(
            y=umbral if umbral_var == "carga_viral" else 5.0,
            line_dash="dot", line_color="crimson",
            annotation_text="umbral", annotation_position="top right",
        )

    fig_s.update_layout(hovermode="x unified", legend_title_text=desglose)
    st.plotly_chart(fig_s, use_container_width=True)

    # --- Mini-KPIs de la serie ---
    m1, m2, m3, m4 = st.columns(4)
    pico_idx  = serie_df.groupby("fecha_reporte")["valor"].sum().idxmax() if desglose != "(ninguno)" else serie_df["valor"].idxmax()
    pico_val  = serie_df.groupby("fecha_reporte")["valor"].sum().max()    if desglose != "(ninguno)" else serie_df["valor"].max()
    pico_fec  = serie_df["fecha_reporte"].iloc[pico_idx] if desglose == "(ninguno)" else pico_idx
    m1.metric("Valor pico", f"{pico_val:,.1f}")
    m2.metric("Fecha pico", str(pico_fec)[:10])
    m3.metric("Períodos", f"{serie_df['fecha_reporte'].nunique()}")
    m4.metric("Último valor", f"{serie_df.sort_values('fecha_reporte').groupby('fecha_reporte')['valor'].sum().iloc[-1]:,.1f}"
              if desglose != "(ninguno)" else f"{serie_df['valor'].iloc[-1]:,.1f}")

    # --- Gráfico de barras apiladas (olas por región) ---
    st.divider()
    st.subheader("🌊 Olas epidémicas — Casos por región y mes")
    olas = (
        dff.assign(mes=dff["fecha_reporte"].dt.to_period("M").dt.to_timestamp())
        .groupby(["mes", "region"], observed=True)
        .size()
        .reset_index(name="casos")
    )
    fig_olas = px.bar(
        olas, x="mes", y="casos", color="region",
        barmode="stack", template=plantilla,
        labels={"mes": "Mes", "casos": "Casos", "region": "Región"},
        title="Distribución mensual de casos por región",
        **color_kwargs(),
    )
    fig_olas.update_layout(hovermode="x unified")
    st.plotly_chart(fig_olas, use_container_width=True)

# ======================================================================
# 8. PESTAÑA — ESTADÍSTICA CUANTITATIVA
# ======================================================================
with tab_cuant:
    st.subheader("Métricas cuantitativas")
    filas = []
    for c in NUMERICAS:
        s   = dff[c].astype(float)
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        atip = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        filas.append({
            "variable": c, "n": s.size,
            "media": s.mean(), "mediana": s.median(),
            "desv_est": s.std(), "error_est": s.sem(),
            "CV %": s.std() / s.mean() * 100 if s.mean() else np.nan,
            "mín": s.min(), "Q1": q1, "Q3": q3, "máx": s.max(), "IQR": iqr,
            "asimetría": stats.skew(s), "curtosis": stats.kurtosis(s),
            "atípicos": atip,
            "IC95 inf": s.mean() - 1.96 * s.sem(),
            "IC95 sup": s.mean() + 1.96 * s.sem(),
        })
    st.dataframe(pd.DataFrame(filas).set_index("variable").round(3), use_container_width=True)
    st.info("CV > 30 % → alta dispersión · Asimetría fuera de ±0.5 → distribución sesgada")

    st.divider()
    st.subheader("Prueba t de Welch")
    c1, c2 = st.columns(2)
    var_t   = c1.selectbox("Variable numérica", NUMERICAS, key="ttest_var")
    grupo_t = c2.selectbox("Agrupación", ["vacunado", "sexo"])
    grupos  = dff.groupby(grupo_t, observed=True)[var_t].apply(list)
    if len(grupos) == 2:
        a, b = grupos.iloc[0], grupos.iloc[1]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        d = (np.mean(a) - np.mean(b)) / np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        c1.metric("t de Welch", f"{t:.3f}")
        c2.metric("p-valor",    f"{p:.2e}")
        st.write(
            f"d de Cohen = **{d:.3f}** · "
            + ("✅ Diferencia significativa (α = 0.05)." if p < 0.05
               else "❌ Sin evidencia de diferencia (α = 0.05).")
        )

# ======================================================================
# 9. PESTAÑA — ESTADÍSTICA CUALITATIVA
# ======================================================================
with tab_cual:
    st.subheader("Tabla de frecuencias")
    var_cual = st.selectbox("Variable categórica", CATEGORICAS)
    freq = (dff[var_cual].value_counts()
            .rename_axis("categoría").reset_index(name="frecuencia"))
    freq["% relativo"]  = (freq["frecuencia"] / freq["frecuencia"].sum() * 100).round(2)
    freq["% acumulado"] = freq["% relativo"].cumsum().round(2)
    st.dataframe(freq, use_container_width=True, hide_index=True)

    p_ent    = freq["frecuencia"] / freq["frecuencia"].sum()
    entropia = -(p_ent * np.log2(p_ent)).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Moda",    str(freq.iloc[0]["categoría"]))
    c2.metric("Categorías", freq.shape[0])
    c3.metric("Entropía de Shannon", f"{entropia:.3f} bits")

    st.divider()
    st.subheader("Tabla de contingencia y χ²")
    c1, c2 = st.columns(2)
    v1 = c1.selectbox("Variable fila",    CATEGORICAS, index=0, key="chi1")
    v2 = c2.selectbox("Variable columna", CATEGORICAS, index=2, key="chi2")
    tabla = pd.crosstab(dff[v1], dff[v2])
    st.dataframe(tabla, use_container_width=True)

    if tabla.shape[0] > 1 and tabla.shape[1] > 1:
        chi2_val, pval, gl, _ = stats.chi2_contingency(tabla)
        n_tot  = tabla.values.sum()
        cramer = np.sqrt(chi2_val / (n_tot * (min(tabla.shape) - 1)))
        a, b, c = st.columns(3)
        a.metric("χ²",         f"{chi2_val:.2f}")
        b.metric("p-valor",    f"{pval:.2e}")
        c.metric("V de Cramér",f"{cramer:.3f}")
        fig_heat = px.imshow(
            tabla, text_auto=True, aspect="auto", template=plantilla,
            color_continuous_scale=paleta if paleta not in ("Plotly","Bold") else "Blues",
            title=f"Heatmap: {v1} × {v2}",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ======================================================================
# 10. PESTAÑA — ANÁLISIS GRÁFICO
# ======================================================================
with tab_graf:
    st.subheader("Distribución univariada")
    c1, c2 = st.columns([1, 2])
    var_dist  = c1.selectbox("Variable", NUMERICAS, key="dist")
    color_por = c1.selectbox("Colorear por", ["(ninguno)"] + CATEGORICAS, key="dist_color")
    tipo      = c1.radio("Tipo", ["Histograma", "Caja", "Violín", "ECDF"])
    mark_umb  = c1.checkbox("Línea de umbral", value=True)

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

    if mark_umb and var_dist == umbral_var:
        etiq = f"umbral = {umbral:.2f}"
        if tipo in ("Histograma", "ECDF"):
            fig.add_vline(x=umbral, line_dash="dash", line_color="crimson", annotation_text=etiq)
        else:
            fig.add_hline(y=umbral, line_dash="dash", line_color="crimson", annotation_text=etiq)
    c2.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Composición por categoría")
    c1, c2 = st.columns(2)
    var_bar = c1.selectbox("Variable de barras", CATEGORICAS, key="bar")
    agrupar = c2.selectbox("Desagregar por", ["(ninguno)"] + CATEGORICAS, key="bar_grp")
    modo    = c1.radio("Modo", ["group", "stack", "relative"], horizontal=True)

    conteo = (
        dff.groupby([var_bar] + ([] if agrupar == "(ninguno)" else [agrupar]), observed=True)
        .size().reset_index(name="casos")
    )
    fig_bar = px.bar(conteo, x=var_bar, y="casos",
                     color=None if agrupar == "(ninguno)" else agrupar,
                     barmode=modo, text_auto=True, template=plantilla,
                     opacity=opacidad, **color_kwargs())
    st.plotly_chart(fig_bar, use_container_width=True)

# ======================================================================
# 11. PESTAÑA — RELACIONES ENTRE VARIABLES
# ======================================================================
with tab_rel:
    st.subheader("Dispersión bivariada")
    c1, c2, c3, c4 = st.columns(4)
    x      = c1.selectbox("Eje X", NUMERICAS, index=0)
    y      = c2.selectbox("Eje Y", NUMERICAS, index=1)
    col_sc = c3.selectbox("Color", ["(ninguno)", "supera_umbral"] + CATEGORICAS)
    tend   = c4.selectbox("Tendencia", ["ninguna", "ols", "lowess"])

    muestra = dff.sample(min(3000, len(dff)), random_state=1)
    fig_sc  = px.scatter(
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

    r,   p_r = stats.pearsonr(dff[x], dff[y])
    rho, p_s = stats.spearmanr(dff[x], dff[y])
    a, b = st.columns(2)
    a.metric("Pearson r",   f"{r:.3f}",   help=f"p = {p_r:.2e}")
    b.metric("Spearman ρ",  f"{rho:.3f}", help=f"p = {p_s:.2e}")

    st.divider()
    st.subheader("Matriz de correlación")
    metodo = st.radio("Método", ["pearson", "spearman", "kendall"], horizontal=True)
    corr   = dff[NUMERICAS].corr(method=metodo).round(3)
    fig_c  = px.imshow(corr, text_auto=True, zmin=-1, zmax=1,
                       color_continuous_scale="RdBu_r", template=plantilla)
    st.plotly_chart(fig_c, use_container_width=True)
