"""Lags estacionales (52 y 104 semanas) y ventana deslizante vs expansiva, walk-forward en 6 folds."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
import exp_lgb as E
p = E.p.sort_values(['BARRIO', 'TW']).reset_index(drop=True)
y = p.target.where(p.target >= 0).astype(float)          # periodo sin etiqueta -> NaN
g = y.groupby(p.BARRIO)
W = 7 * 24                                                # horas en una semana
p['lag52'] = g.shift(52 * W); p['lag104'] = g.shift(104 * W)
p['lag52_s'] = pd.concat([g.shift(k * W) for k in range(50, 55)], axis=1).mean(axis=1)    # semanas 50-54, misma hora y dia
p['lag104_s'] = pd.concat([g.shift(k * W) for k in range(102, 107)], axis=1).mean(axis=1)
dia = p.assign(d=p.TW.dt.normalize(), y=y).groupby(['BARRIO', 'd']).y.sum(min_count=1)
d364 = dia.groupby(level=0).shift(364).rename('lag52_dia')                                 # accidentes del barrio ese dia hace 52 semanas
p = p.merge(d364.reset_index(), left_on=['BARRIO', p.TW.dt.normalize()], right_on=['BARRIO', 'd'], how='left').drop(columns=['d', 'key_1'], errors='ignore')

CORTES = ['2018-07-01', '2018-08-01', '2018-10-01', '2018-12-01', '2019-01-01', '2019-03-01']   # 6 folds de 5 meses
SETS = {'cal': E.CAL, 'cal+lag52': E.CAL + ['lag52'], 'cal+lag52+lag104': E.CAL + ['lag52', 'lag104'],
        'cal+lags_suav': E.CAL + ['lag52_s', 'lag104_s', 'lag52_dia']}
VENT = {'expansiva': None, 'desl_6m': 6, 'desl_12m': 12, 'desl_18m': 18}
rows = []
for c in CORTES:
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5)
    va = p[(p.TW >= a) & (p.TW < b)]
    for vn, m in VENT.items():
        ini = pd.Timestamp('2017-01-01') if m is None else a - pd.DateOffset(months=m)
        tr = p[(p.TW >= ini) & (p.TW < a) & (p.target >= 0)]
        for sn, F in SETS.items():
            if vn != 'expansiva' and sn not in ('cal', 'cal+lags_suav'):
                continue
            mdl = lgb.LGBMClassifier(**E.base, verbose=-1).fit(tr[F], tr.target)
            rows.append(dict(corte=c[:7], ventana=vn, variables=sn, auc=auc(va.target, mdl.predict_proba(va[F])[:, 1])))
    print(c, 'listo', flush=True)
r = pd.DataFrame(rows); r.to_csv('res_lags_ventana.csv', index=False)
t = r.pivot_table(index=['ventana', 'variables'], columns='corte', values='auc')
t['promedio'] = t.mean(axis=1)
print(t.round(4).sort_values('promedio', ascending=False).to_string())
