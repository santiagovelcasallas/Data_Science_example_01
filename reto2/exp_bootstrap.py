"""Bootstrap: (1) bagging de LightGBM con bootstrap por semanas y por filas; (2) IC del AUC remuestreando dias de validacion."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
import exp_lgb as E
p = E.p; p['sem'] = p.TW.dt.to_period('W').astype(str)
CORTES = ['2018-07-01', '2018-08-01', '2018-10-01', '2018-12-01', '2019-01-01', '2019-03-01']
NB = 20
def fit(d): return lgb.LGBMClassifier(**E.base, verbose=-1).fit(d[E.CAL], d.target)
rows, ic = [], []
for c in CORTES:
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5)
    tr = p[(p.TW < a) & (p.target >= 0)]; va = p[(p.TW >= a) & (p.TW < b)]
    s = {'lgb (base)': fit(tr).predict_proba(va[E.CAL])[:, 1]}
    rng = np.random.default_rng(0); sems = tr.sem.unique(); idx = tr.groupby('sem').indices
    bw = bf = 0
    for i in range(NB):
        pick = rng.choice(sems, len(sems), replace=True)
        bw = bw + fit(tr.iloc[np.concatenate([idx[w] for w in pick])]).predict_proba(va[E.CAL])[:, 1]
        bf = bf + fit(tr.iloc[rng.integers(0, len(tr), len(tr))]).predict_proba(va[E.CAL])[:, 1]
    s['bagging 20 (bootstrap por semanas)'] = bw / NB; s['bagging 20 (bootstrap por filas)'] = bf / NB
    for n, v in s.items(): rows.append(dict(corte=c[:7], modelo=n, auc=auc(va.target, v)))
    # IC 95 % del AUC del modelo base remuestreando dias completos del periodo de validacion
    d = va.TW.dt.normalize().values; dias = np.unique(d); di = pd.Series(np.arange(len(va))).groupby(d).indices
    y = va.target.values; sc = s['lgb (base)']; aucs = []
    for _ in range(1000):
        j = np.concatenate([di[x] for x in rng.choice(dias, len(dias), replace=True)])
        aucs.append(auc(y[j], sc[j]))
    ic.append(dict(corte=c[:7], auc=auc(y, sc), ic_2_5=np.percentile(aucs, 2.5), ic_97_5=np.percentile(aucs, 97.5),
                   maximo_de_1000=np.max(aucs), prob_auc_mayor_085=np.mean(np.array(aucs) >= 0.85)))
    print(c, 'listo', flush=True)
t = pd.DataFrame(rows).pivot_table(index='modelo', columns='corte', values='auc'); t['promedio'] = t.mean(axis=1)
print('Tabla 1. Bagging\n', t.round(4).to_string())
print('Tabla 2. IC 95% del AUC (bootstrap de dias)\n', pd.DataFrame(ic).round(4).to_string(index=False))
