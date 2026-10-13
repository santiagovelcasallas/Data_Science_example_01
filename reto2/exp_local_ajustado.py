"""Un LightGBM por barrio con complejidad elegida por barrio (validacion temporal interna), predicciones unidas. 6 cortes."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc, log_loss
import exp_lgb as E
p = E.p; CT = E.CAL + ['t']; CL = [c for c in CT if c != 'barrio']
COMUN = dict(learning_rate=0.02, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5, verbose=-1)
NIVELES = {'muy simple': dict(num_leaves=3, min_child_samples=200, n_estimators=200),
           'simple': dict(num_leaves=7, min_child_samples=100, n_estimators=300),
           'medio': dict(num_leaves=15, min_child_samples=50, n_estimators=400),
           'complejo': dict(num_leaves=31, min_child_samples=20, n_estimators=600)}
CORTES = ['2018-07-01', '2018-08-01', '2018-10-01', '2018-12-01', '2019-01-01', '2019-03-01']
fit = lambda d, n: lgb.LGBMClassifier(**COMUN, **NIVELES[n]).fit(d[CL], d.target)
rows, elec, gana = [], [], []
for c in CORTES:
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5); ai = a - pd.DateOffset(months=5)
    tr = p[(p.TW < a) & (p.target >= 0)]; va = p[(p.TW >= a) & (p.TW < b)].copy()
    va['LGB global'] = lgb.LGBMClassifier(**E.base, verbose=-1).fit(tr[CT], tr.target).predict_proba(va[CT])[:, 1]
    va['local fijo (simple)'] = np.nan; va['local ajustado por barrio'] = np.nan
    for bar in va.BARRIO.unique():
        t_ = tr[tr.BARRIO == bar]; v_ = va.BARRIO == bar
        ti, vi = t_[t_.TW < ai], t_[t_.TW >= ai]                       # validacion temporal interna
        ll = {n: log_loss(vi.target, fit(ti, n).predict_proba(vi[CL])[:, 1], labels=[0, 1]) for n in NIVELES}
        mejor = min(ll, key=ll.get); elec.append(dict(corte=c[:7], barrio=bar, nivel=mejor))
        va.loc[v_, 'local ajustado por barrio'] = fit(t_, mejor).predict_proba(va.loc[v_, CL])[:, 1]
        va.loc[v_, 'local fijo (simple)'] = fit(t_, 'simple').predict_proba(va.loc[v_, CL])[:, 1]
        g = va[v_]
        if g.target.sum() > 0:
            gana.append(dict(corte=c[:7], barrio=bar, gana_local=auc(g.target, g['local ajustado por barrio']) > auc(g.target, g['LGB global'])))
    for m in ['LGB global', 'local fijo (simple)', 'local ajustado por barrio']:
        rows.append(dict(corte=c[:7], modelo=m, auc=auc(va.target, va[m])))
    print(c, 'listo', flush=True)
t = pd.DataFrame(rows).pivot_table(index='modelo', columns='corte', values='auc'); t['promedio'] = t.mean(axis=1)
print('Tabla 1. AUC global (22 barrios unidos)'); print(t.round(4).sort_values('promedio', ascending=False).to_string())
e = pd.DataFrame(elec)
print('\nTabla 2. Complejidad elegida (numero de barrios por corte)')
print(e.pivot_table(index='nivel', columns='corte', values='barrio', aggfunc='count').reindex(list(NIVELES)).fillna(0).astype(int).to_string())
est = e.groupby('barrio').nivel.nunique()
print('barrios que eligen siempre el mismo nivel en los 6 cortes:', (est == 1).sum(), 'de', len(est))
g = pd.DataFrame(gana)
print('\nTabla 3. Barrios donde el local ajustado supera al global (AUC dentro del barrio)')
print(g.groupby('corte').gana_local.sum().astype(int).to_string(), '\n(de 22 barrios en cada corte)')
