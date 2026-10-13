"""Por que Prophet no suma: escasez de los conteos diarios y aporte de festivos dentro de LightGBM."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
import exp_lgb as E
p = E.p; t = p[p.target >= 0].copy(); t['d'] = t.TW.dt.normalize()
dia = t.groupby(['BARRIO', 'd']).n_acc.sum()
print('conteo diario por barrio: media=%.2f  dias con 0 accidentes=%.0f%%' % (dia.mean(), 100 * (dia == 0).mean()))
# efecto festivo sobre el perfil horario: tasa por hora en festivo vs laboral
t['tipo'] = np.where(t.festivo == 1, 'festivo', np.where(t.dow >= 5, 'fin_semana', 'laboral'))
perf = t.groupby(['tipo', 'h']).target.mean().unstack(0)
print((perf.loc[[7, 8, 12, 18, 23]] * 100).round(2).rename_axis('hora').to_string(), '\n(% de horas con accidente)')
a, b = E.FOLDS['A']
tr = p[(p.TW < a) & (p.target >= 0)]; va = p[(p.TW >= a) & (p.TW < b)]
for name, F in [('con festivos', E.CAL), ('sin festivos', [c for c in E.CAL if c not in ('festivo', 'tipo_dia', 'vispera', 'lunes_fest')])]:
    m = lgb.LGBMClassifier(**E.base, verbose=-1).fit(tr[F], tr.target)
    print(name, 'AUC fold A = %.4f' % auc(va.target, m.predict_proba(va[F])[:, 1]))
