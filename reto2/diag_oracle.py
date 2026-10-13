"""Cuanta senal hay en cada nivel: base barrio x hora x tipo_dia, multiplicada por niveles 'oraculo'."""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score as auc
p = pd.read_pickle('data/panel.pkl'); p = p[p.target >= 0].copy()
p['h'] = p.TW.dt.hour; p['dow'] = p.TW.dt.dayofweek; p['d'] = p.TW.dt.normalize()
folds = {'A': ('2018-08-01', '2019-01-01'), 'B': ('2019-03-01', '2019-08-01')}
for f, (a, b) in folds.items():
    tr = p[p.TW < a]; va = p[(p.TW >= a) & (p.TW < b)].copy()
    base = tr.groupby(['BARRIO', 'h']).target.mean()
    prof = tr.groupby(['dow', 'h']).target.mean() / tr.target.mean()
    va['base'] = va.set_index(['BARRIO', 'h']).index.map(base).values * va.set_index(['dow', 'h']).index.map(prof).values
    r = {'base': auc(va.target, va.base)}
    for lvl, key in [('dia', ['d']), ('dia_barrio', ['d', 'BARRIO']), ('semana', [va.TW.dt.isocalendar().week]),
                     ('mes_barrio', [va.TW.dt.month, 'BARRIO']), ('hora_global', ['TW']), ('dia_x_franja', ['d', va.h // 6])]:
        g = va.groupby(key).n_acc.transform('sum') if lvl != 'dia_barrio' else va.groupby(key).n_acc.transform('sum')
        # oraculo suavizado: nivel observado (+1) / esperado
        e = va.groupby(key).base.transform('sum')
        r[lvl] = auc(va.target, va.base * (g - va.n_acc + 1) / (e - va.base + 1))
    print(f, {k: round(v, 4) for k, v in r.items()})
