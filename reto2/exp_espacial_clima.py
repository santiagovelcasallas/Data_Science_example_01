"""Variables espaciales (desde raw, solo antes del corte) y clima diario. 6 cortes, ventana expansiva."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
import exp_lgb as E
p = E.p.sort_values(['BARRIO', 'TW']).reset_index(drop=True)
p['d'] = p.TW.dt.normalize(); p['trim'] = p.TW.dt.year * 10 + p.TW.dt.quarter
raw = pd.read_pickle('data/raw.pkl')
raw = raw[raw.Lon > -75.62]                                   # quita los 5 geocodificados en San Antonio de Prado

# --- clima diario por barrio y lluvia previa
gb = p.groupby(['BARRIO', 'd'])
cd = gb.agg(lluvia_dia=('precipIntensity', 'sum'), lluvia_max_dia=('precipIntensity', 'max'), temp_dia=('temperature', 'mean'),
            hum_dia=('humidity', 'mean'), nubes_dia=('cloudCover', 'mean'), visib_min_dia=('visibility', 'min')).reset_index()
p = p.merge(cd, on=['BARRIO', 'd'], how='left')
pr = p.groupby('BARRIO').precipIntensity
for k in (1, 3, 6):
    p[f'lluvia_prev_{k}h'] = pr.transform(lambda s: s.shift(1).rolling(k, min_periods=1).sum())
CLIMA_DIA = list(cd.columns[2:]); LLUVIA_PREV = ['lluvia_prev_1h', 'lluvia_prev_3h', 'lluvia_prev_6h']

def espacial_static(a):
    r = raw[raw.TW < a]
    f = r.groupby('BARRIO').agg(lat=('Lat', 'mean'), lon=('Lon', 'mean'),
                                pct_inters=('DISENO', lambda s: (s == 'Interseccion').mean() + (s == 'Intersección').mean()),
                                pct_glorieta=('DISENO', lambda s: (s == 'Glorieta').mean()),
                                pct_heridos=('GRAVEDAD', lambda s: s.str.contains('HERIDO', na=False).mean()),
                                pct_atropello=('CLASE', lambda s: (s == 'Atropello').mean()),
                                n_direcciones=('DIRECCION', 'nunique'))
    return f

def vecinos(src, dst, cent, k=3, m=100):
    """Tasa suavizada en hora x tipo_dia de los k barrios mas cercanos (sin el propio)."""
    xy = cent[['lat', 'lon']].values; names = cent.index.values
    dist = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1); np.fill_diagonal(dist, np.inf)
    nb = {names[i]: names[np.argsort(dist[i])[:k]] for i in range(len(names))}
    st = src.groupby(['BARRIO', 'h', 'tipo_dia']).target.agg(['sum', 'count'])
    gl = src.groupby(['h', 'tipo_dia']).target.mean()
    out = np.empty(len(dst))
    for b, idx in dst.groupby('BARRIO').indices.items():
        s = st.loc[st.index.get_level_values(0).isin(nb[b])].groupby(level=[1, 2]).sum()
        sub = dst.iloc[idx]; key = pd.MultiIndex.from_arrays([sub.h, sub.tipo_dia])
        ss = s['sum'].reindex(key).fillna(0).values; nn = s['count'].reindex(key).fillna(0).values
        out[idx] = (ss + m * gl.reindex(key).values) / (nn + m)
    return out

CORTES = ['2018-07-01', '2018-08-01', '2018-10-01', '2018-12-01', '2019-01-01', '2019-03-01']
ESP = ['lat', 'lon', 'pct_inters', 'pct_glorieta', 'pct_heridos', 'pct_atropello', 'n_direcciones']
SETS = {'cal': E.CAL, 'cal+esp_estaticas': E.CAL + ESP, 'cal+tasa_vecinos': E.CAL + ['tasa_vecinos'],
        'cal+clima_dia': E.CAL + CLIMA_DIA, 'cal+clima_dia+lluvia_prev': E.CAL + CLIMA_DIA + LLUVIA_PREV,
        'cal+todo': E.CAL + ESP + ['tasa_vecinos'] + CLIMA_DIA + LLUVIA_PREV}
rows = []
for c in CORTES:
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5)
    tr = p[(p.TW < a) & (p.target >= 0)].copy(); va = p[(p.TW >= a) & (p.TW < b)].copy()
    est = espacial_static(a)
    tr = tr.join(est, on='BARRIO'); va = va.join(est, on='BARRIO')
    tv = np.empty(len(tr))
    for q in tr.trim.unique():
        msk = (tr.trim == q).values; tv[msk] = vecinos(tr[~msk], tr[msk], est)
    tr['tasa_vecinos'] = tv; va['tasa_vecinos'] = vecinos(tr, va, est)
    for sn, F in SETS.items():
        mdl = lgb.LGBMClassifier(**E.base, verbose=-1).fit(tr[F], tr.target)
        rows.append(dict(corte=c[:7], variables=sn, auc=auc(va.target, mdl.predict_proba(va[F])[:, 1])))
    print(c, 'listo', flush=True)
r = pd.DataFrame(rows); r.to_csv('res_espacial_clima.csv', index=False)
t = r.pivot_table(index='variables', columns='corte', values='auc'); t['promedio'] = t.mean(axis=1)
print(t.round(4).sort_values('promedio', ascending=False).to_string())
