"""Variantes de LightGBM (solo calendario): capacidad, regularizacion y objetivo Poisson."""
import numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
import exp_lgb as E  # reutiliza panel y folds
CAL = E.CAL
def run(params, y='target', feats=CAL):
    r = {}
    for f, (a, b) in E.FOLDS.items():
        tr = E.p[(E.p.TW < a) & (E.p.target >= 0)]; va = E.p[(E.p.TW >= a) & (E.p.TW < b)]
        M = lgb.LGBMRegressor if params.get('objective') == 'poisson' else lgb.LGBMClassifier
        m = M(**params, verbose=-1).fit(tr[feats], tr[y])
        s = m.predict(va[feats]) if M is lgb.LGBMRegressor else m.predict_proba(va[feats])[:, 1]
        r[f] = auc(va.target, s)
    return r
b = E.base
for name, pr, y in [('lento', {**b, 'n_estimators': 1200, 'learning_rate': 0.01}, 'target'),
                    ('hojas7', {**b, 'num_leaves': 7, 'n_estimators': 800}, 'target'),
                    ('hojas31', {**b, 'num_leaves': 31, 'min_child_samples': 1000}, 'target'),
                    ('poisson', {**b, 'objective': 'poisson'}, 'n_acc'),
                    ('sin_doy', b, 'target')]:
    feats = [c for c in CAL if c != 'doy'] if name == 'sin_doy' else CAL
    r = run(pr, y, feats)
    print(name, {k: round(v, 4) for k, v in r.items()}, 'AB=', round((r['A'] + r['B']) / 2, 4), 'all=', round(np.mean(list(r.values())), 4), flush=True)
