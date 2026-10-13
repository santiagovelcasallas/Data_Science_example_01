"""TE fuera de muestra de barrio x mes y barrio x tipo_dia, como variables de LightGBM. 6 cortes, ventana expansiva."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
import exp_lgb as E
p = E.p; p['trim'] = p.TW.dt.year * 10 + p.TW.dt.quarter
CORTES = ['2018-07-01', '2018-08-01', '2018-10-01', '2018-12-01', '2019-01-01', '2019-03-01']
INTER = {'te_barrio_mes': 'mes', 'te_barrio_tipodia': 'tipo_dia'}

def encode(src, dst, col, m):
    """TE suavizado de BARRIO x col calculado en src y aplicado a dst. Prior = tasa barrio * efecto col / tasa global."""
    g = src.target.mean()
    pb = src.groupby('BARRIO').target.mean(); pc = src.groupby(col).target.mean()
    st = src.groupby(['BARRIO', col]).target.agg(['sum', 'count'])
    k = pd.MultiIndex.from_arrays([dst.BARRIO, dst[col]])
    s = st['sum'].reindex(k).fillna(0).values; n = st['count'].reindex(k).fillna(0).values
    prior = dst.BARRIO.map(pb).values * dst[col].map(pc).values / g
    return (s + m * prior) / (n + m) / prior            # razon frente al prior (1 = sin efecto propio)

def te_oof(tr, va, col, m):
    out = np.empty(len(tr))
    for q in tr.trim.unique():                           # fuera de muestra por trimestre
        msk = (tr.trim == q).values
        out[msk] = encode(tr[~msk], tr[msk], col, m)
    return out, encode(tr, va, col, m)

rows = []
for c in CORTES:
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5)
    tr = p[(p.TW < a) & (p.target >= 0)].copy(); va = p[(p.TW >= a) & (p.TW < b)].copy()
    for m in (50, 300):
        for n, col in INTER.items():
            tr[f'{n}_{m}'], va[f'{n}_{m}'] = te_oof(tr, va, col, m)
    sets = {'cal': E.CAL}
    for m in (50, 300):
        sets[f'cal+te_barrio_mes (m={m})'] = E.CAL + [f'te_barrio_mes_{m}']
        sets[f'cal+te_barrio_tipodia (m={m})'] = E.CAL + [f'te_barrio_tipodia_{m}']
        sets[f'cal+ambos (m={m})'] = E.CAL + [f'te_barrio_mes_{m}', f'te_barrio_tipodia_{m}']
    for sn, F in sets.items():
        mdl = lgb.LGBMClassifier(**E.base, verbose=-1).fit(tr[F], tr.target)
        rows.append(dict(corte=c[:7], variables=sn, auc=auc(va.target, mdl.predict_proba(va[F])[:, 1])))
    print(c, 'listo', flush=True)
r = pd.DataFrame(rows); r.to_csv('res_te.csv', index=False)
t = r.pivot_table(index='variables', columns='corte', values='auc'); t['promedio'] = t.mean(axis=1)
print(t.round(4).sort_values('promedio', ascending=False).to_string())
