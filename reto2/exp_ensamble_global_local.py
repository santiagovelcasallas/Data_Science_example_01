"""Ensamble del LGB global con el local estandarizado al nivel del global. p = w*global + (1-w)*local_std. 6 cortes."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
import exp_lgb as E
p = E.p; CT = E.CAL + ['t']; CL = [c for c in CT if c != 'barrio']
LOC = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=100, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.7, reg_lambda=5, verbose=-1)
CORTES = ['2018-07-01', '2018-08-01', '2018-10-01', '2018-12-01', '2019-01-01', '2019-03-01']
W = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))
rows = []
for c in CORTES:
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5)
    tr = p[(p.TW < a) & (p.target >= 0)]; va = p[(p.TW >= a) & (p.TW < b)].copy()
    g = lgb.LGBMClassifier(**E.base, verbose=-1).fit(tr[CT], tr.target).predict_proba(va[CT])[:, 1]; va['g'] = g
    va['l'] = np.nan
    for bar in va.BARRIO.unique():
        t_ = tr[tr.BARRIO == bar]; v_ = (va.BARRIO == bar).values
        s = lgb.LGBMClassifier(**LOC).fit(t_[CL], t_.target).predict_proba(va.loc[v_, CL])[:, 1]
        va.loc[v_, 'l'] = s * va.loc[v_, 'g'].mean() / s.mean()        # local estandarizado al nivel del global
    for w in W:
        rows.append(dict(corte=c[:7], tipo='promedio de probabilidades', w=w, auc=auc(va.target, w * va.g + (1 - w) * va.l)))
        rows.append(dict(corte=c[:7], tipo='promedio en escala logit', w=w, auc=auc(va.target, w * lg(va.g) + (1 - w) * lg(va.l))))
    print(c, 'listo', flush=True)
r = pd.DataFrame(rows)
t = r.pivot_table(index=['tipo', 'w'], columns='corte', values='auc'); t['promedio'] = t.mean(axis=1)
print('Tabla 1. AUC segun el peso w del global (w=1: solo global; w=0: solo local estandarizado)')
print(t.round(4).sort_index(ascending=[True, False]).to_string())
aj = r[r.corte.isin(['2018-07', '2018-10'])].groupby(['tipo', 'w']).auc.mean(); best = aj.idxmax()
pr = r[(r.corte == '2019-03')].set_index(['tipo', 'w']).auc
print(f'\nTabla 2. w elegido en 2018-07 y 2018-10: {best}  (AUC ajuste {aj[best]:.4f})')
print(f'  prueba limpia 2019-03: ensamble {pr[best]:.4f}  vs  solo global {pr[(best[0], 1.0)]:.4f}')
