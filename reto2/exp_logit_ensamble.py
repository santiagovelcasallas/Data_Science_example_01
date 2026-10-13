"""Logistica L2 con interacciones one-hot y ensamble por rango con LightGBM. 6 cortes, ventana expansiva."""
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score as auc
import exp_lgb as E
p = E.p
p['b_h'] = p.BARRIO + '_' + p.h.astype(str); p['b_t'] = p.BARRIO + '_' + p.tipo_dia.astype(str)
p['h_t'] = p.h.astype(str) + '_' + p.tipo_dia.astype(str); p['b_h_t'] = p.b_h + '_' + p.tipo_dia.astype(str)
BASE = ['BARRIO', 'h', 'tipo_dia', 'mes', 'festivo', 'vispera', 'lunes_fest', 'quincena', 'dic_ene', 'feria', 'b_h', 'b_t', 'h_t']
LOGITS = {'logit C=0.03': (BASE, 0.03), 'logit C=0.3': (BASE, 0.3),
          'logit+triple C=0.03': (BASE + ['b_h_t'], 0.03), 'logit+triple C=0.3': (BASE + ['b_h_t'], 0.3)}
CORTES = ['2018-07-01', '2018-08-01', '2018-10-01', '2018-12-01', '2019-01-01', '2019-03-01']
rk = lambda s: rankdata(s) / len(s)
rows = []
for c in CORTES:
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5)
    tr = p[(p.TW < a) & (p.target >= 0)]; va = p[(p.TW >= a) & (p.TW < b)]
    s = {}
    s['lgb (cal)'] = lgb.LGBMClassifier(**E.base, verbose=-1).fit(tr[E.CAL], tr.target).predict_proba(va[E.CAL])[:, 1]
    for n, (cols, C) in LOGITS.items():
        enc = OneHotEncoder(handle_unknown='ignore').fit(tr[cols].astype(str))
        m = LogisticRegression(C=C, max_iter=2000).fit(enc.transform(tr[cols].astype(str)), tr.target)
        s[n] = m.predict_proba(enc.transform(va[cols].astype(str)))[:, 1]
        s[f'rango(lgb, {n})'] = 0.5 * rk(s['lgb (cal)']) + 0.5 * rk(s[n])
    for n, v in s.items():
        rows.append(dict(corte=c[:7], modelo=n, auc=auc(va.target, v)))
    print(c, 'listo', flush=True)
r = pd.DataFrame(rows); r.to_csv('res_logit.csv', index=False)
t = r.pivot_table(index='modelo', columns='corte', values='auc'); t['promedio'] = t.mean(axis=1)
print(t.round(4).sort_values('promedio', ascending=False).to_string())
