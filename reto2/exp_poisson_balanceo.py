"""Poisson sobre n_acc, SMOTENC y EasyEnsemble con LightGBM (barrio + calendario). 6 cortes, ventana expansiva."""
import numpy as np, pandas as pd, lightgbm as lgb
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import roc_auc_score as auc
import exp_lgb as E
p = E.p; F = E.CAL
CAT = [i for i, c in enumerate(F) if c != 'doy']                  # todo categorico salvo dia del ano
CORTES = ['2018-07-01', '2018-08-01', '2018-10-01', '2018-12-01', '2019-01-01', '2019-03-01']
X = lambda d: d[F].assign(barrio=d.barrio.cat.codes)               # SMOTENC necesita numeros
def clf(): return lgb.LGBMClassifier(**E.base, verbose=-1)

def easy(tr, va, ratio, n=10):
    pos, neg = tr[tr.target == 1], tr[tr.target == 0]; s = 0
    for i in range(n):
        sub = pd.concat([pos, neg.sample(len(pos) * ratio, random_state=i)])
        s = s + clf().fit(X(sub), sub.target).predict_proba(X(va))[:, 1]
    return s / n

rows = []
for c in CORTES:
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5)
    tr = p[(p.TW < a) & (p.target >= 0)]; va = p[(p.TW >= a) & (p.TW < b)]
    s = {'lgb binario (base)': clf().fit(X(tr), tr.target).predict_proba(X(va))[:, 1],
         'lgb poisson n_acc': lgb.LGBMRegressor(**E.base, objective='poisson', verbose=-1).fit(X(tr), tr.n_acc).predict(X(va))}
    for r in (0.1, 0.5):
        Xs, ys = SMOTENC(categorical_features=CAT, sampling_strategy=r, random_state=0).fit_resample(X(tr), tr.target)
        s[f'smote {int(r*100)}%'] = clf().fit(Xs, ys).predict_proba(X(va))[:, 1]
    s['easyensemble 1:1'] = easy(tr, va, 1); s['easyensemble 1:9'] = easy(tr, va, 9)
    for n, v in s.items():
        rows.append(dict(corte=c[:7], modelo=n, auc=auc(va.target, v)))
    print(c, 'listo', flush=True)
r = pd.DataFrame(rows); r.to_csv('res_poisson_balanceo.csv', index=False)
t = r.pivot_table(index='modelo', columns='corte', values='auc'); t['promedio'] = t.mean(axis=1)
print(t.round(4).sort_values('promedio', ascending=False).to_string())
