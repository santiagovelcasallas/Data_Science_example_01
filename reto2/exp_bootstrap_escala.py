"""Bootstrap para igualar el tamano de los barrios y estandarizacion de la escala de los modelos locales. 6 cortes."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auc
import exp_lgb as E
p = E.p; CT = E.CAL + ['t']; CL = [c for c in CT if c != 'barrio']
LOC = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=100, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.7, reg_lambda=5, verbose=-1)
CORTES = ['2018-07-01', '2018-08-01', '2018-10-01', '2018-12-01', '2019-01-01', '2019-03-01']
loc = lambda d: lgb.LGBMClassifier(**LOC).fit(d[CL], d.target)
logit_ = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
rows = []
for c in CORTES:
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5); ai = a - pd.DateOffset(months=5)
    tr = p[(p.TW < a) & (p.target >= 0)]; va = p[(p.TW >= a) & (p.TW < b)].copy()
    rng = np.random.default_rng(0)
    va['LGB global'] = lgb.LGBMClassifier(**E.base, verbose=-1).fit(tr[CT], tr.target).predict_proba(va[CT])[:, 1]
    # 1. global con bootstrap balanceado por barrio (todos con el mismo numero de filas)
    n = tr.groupby('BARRIO').size().max(); idx = tr.groupby('BARRIO').indices
    trb = tr.iloc[np.concatenate([rng.choice(ix, n, replace=True) for ix in idx.values()])]
    va['global, barrios balanceados (bootstrap)'] = lgb.LGBMClassifier(**E.base, verbose=-1).fit(trb[CT], trb.target).predict_proba(va[CT])[:, 1]
    for col in ['local', 'local, bootstrap igualado', 'local estandarizado al nivel del global', 'local calibrado (Platt por barrio)', 'local z-score por barrio']:
        va[col] = np.nan
    for bar in va.BARRIO.unique():
        t_ = tr[tr.BARRIO == bar]; v_ = (va.BARRIO == bar).values
        s = loc(t_).predict_proba(va.loc[v_, CL])[:, 1]; va.loc[v_, 'local'] = s
        # 2. bootstrap hasta el tamano del barrio mas grande
        tb = t_.iloc[rng.integers(0, len(t_), n)]
        va.loc[v_, 'local, bootstrap igualado'] = loc(tb).predict_proba(va.loc[v_, CL])[:, 1]
        # 3. mismo orden local, nivel medio del barrio = el que predice el global (solo predicciones, sin etiquetas)
        va.loc[v_, 'local estandarizado al nivel del global'] = s * va.loc[v_, 'LGB global'].mean() / s.mean()
        # 4. Platt por barrio: se ajusta en la validacion interna (ultimos 5 meses antes del corte)
        ti, vi = t_[t_.TW < ai], t_[t_.TW >= ai]
        cal = LogisticRegression().fit(logit_(loc(ti).predict_proba(vi[CL])[:, 1]).reshape(-1, 1), vi.target)
        va.loc[v_, 'local calibrado (Platt por barrio)'] = cal.predict_proba(logit_(s).reshape(-1, 1))[:, 1]
        # 5. z-score por barrio
        va.loc[v_, 'local z-score por barrio'] = (s - s.mean()) / s.std()
    for m in ['LGB global', 'global, barrios balanceados (bootstrap)', 'local', 'local, bootstrap igualado',
              'local estandarizado al nivel del global', 'local calibrado (Platt por barrio)', 'local z-score por barrio']:
        rows.append(dict(corte=c[:7], modelo=m, auc=auc(va.target, va[m])))
    print(c, 'listo', flush=True)
t = pd.DataFrame(rows).pivot_table(index='modelo', columns='corte', values='auc'); t['promedio'] = t.mean(axis=1)
print(t.round(4).sort_values('promedio', ascending=False).to_string())
