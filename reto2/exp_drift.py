"""Drift por barrio: tendencia t y nivel reciente del barrio (3/6/12 meses antes del corte / historico). 6 cortes."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
import exp_lgb as E
p = E.p
CORTES = ['2018-07-01', '2018-08-01', '2018-10-01', '2018-12-01', '2019-01-01', '2019-03-01']
REC = ['nivel_3m', 'nivel_6m', 'nivel_12m']
SETS = {'cal': E.CAL, 'cal+t': E.CAL + ['t'], 'cal+nivel_reciente': E.CAL + REC, 'cal+t+nivel_reciente': E.CAL + ['t'] + REC}
def nivel(tr, ref, meses, m=20):
    """Tasa del barrio en los ultimos `meses` antes de ref / tasa historica, suavizada (m positivos de prior)."""
    h = tr.groupby('BARRIO').target.mean()
    r = tr[tr.TW >= ref - pd.DateOffset(months=meses)].groupby('BARRIO').target.agg(['sum', 'count'])
    return (r['sum'] + m) / (r['count'] * h + m)
rows = []
for c in CORTES:
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5)
    tr = p[(p.TW < a) & (p.target >= 0)].copy(); va = p[(p.TW >= a) & (p.TW < b)].copy()
    tr['mes_ref'] = tr.TW.dt.to_period('M').dt.to_timestamp()
    for k in (3, 6, 12):
        va[f'nivel_{k}m'] = va.BARRIO.map(nivel(tr, a, k))
        # en entrenamiento, cada mes ve el nivel calculado solo con lo anterior a ese mes (sin fuga)
        vals = {}
        for mref in tr.mes_ref.unique():
            prev = tr[tr.TW < mref]
            if len(prev) and prev.TW.min() <= mref - pd.DateOffset(months=k):
                vals[mref] = nivel(prev, mref, k)
        tr[f'nivel_{k}m'] = [vals[m_][b_] if m_ in vals else np.nan for m_, b_ in zip(tr.mes_ref, tr.BARRIO)]
    for sn, F in SETS.items():
        mdl = lgb.LGBMClassifier(**E.base, verbose=-1).fit(tr[F], tr.target)
        rows.append(dict(corte=c[:7], variables=sn, auc=auc(va.target, mdl.predict_proba(va[F])[:, 1])))
    print(c, 'listo', flush=True)
t = pd.DataFrame(rows).pivot_table(index='variables', columns='corte', values='auc'); t['promedio'] = t.mean(axis=1)
print(t.round(4).sort_values('promedio', ascending=False).to_string())
