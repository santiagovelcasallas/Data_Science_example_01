"""Optuna sobre LightGBM (barrio + calendario). Ajusta en cortes 2018-07 y 2018-10; 2019-03 queda como prueba limpia."""
import json, numpy as np, pandas as pd, lightgbm as lgb, optuna
from sklearn.metrics import roc_auc_score as auc
import exp_lgb as E
optuna.logging.set_verbosity(optuna.logging.WARNING)
p = E.p
def split(c):
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5)
    return p[(p.TW < a) & (p.target >= 0)], p[(p.TW >= a) & (p.TW < b)]
AJUSTE = ['2018-07-01', '2018-10-01']
TODOS = ['2018-07-01', '2018-08-01', '2018-10-01', '2018-12-01', '2019-01-01', '2019-03-01']
DATA = {c: split(c) for c in TODOS}
def score(params, c):
    tr, va = DATA[c]
    m = lgb.LGBMClassifier(**params, subsample_freq=1, verbose=-1).fit(tr[E.CAL], tr.target)
    return auc(va.target, m.predict_proba(va[E.CAL])[:, 1])
def objetivo(t):
    pr = dict(num_leaves=t.suggest_int('num_leaves', 4, 63, log=True),
              min_child_samples=t.suggest_int('min_child_samples', 50, 3000, log=True),
              learning_rate=t.suggest_float('learning_rate', 0.005, 0.1, log=True),
              n_estimators=t.suggest_int('n_estimators', 100, 1500, log=True),
              subsample=t.suggest_float('subsample', 0.5, 1.0), colsample_bytree=t.suggest_float('colsample_bytree', 0.3, 1.0),
              reg_lambda=t.suggest_float('reg_lambda', 1e-3, 100, log=True),
              min_split_gain=t.suggest_float('min_split_gain', 0, 0.5))
    v = np.mean([score(pr, c) for c in AJUSTE])
    print(f'trial {t.number}: {v:.4f}', flush=True)
    return v
st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
st.enqueue_trial({k: v for k, v in E.base.items() if k != 'subsample_freq'} | {'min_split_gain': 0.0})   # trial 0 = parametros base
st.optimize(objetivo, n_trials=40)
best = st.best_params; json.dump(best, open('optuna_best.json', 'w'), indent=1)
print('mejores parametros', best)
base = {k: v for k, v in E.base.items() if k != 'subsample_freq'}
rows = [dict(corte=c[:7], modelo=n, auc=score(pr, c)) for c in TODOS for n, pr in [('lgb base', base), ('lgb optuna', best)]]
t = pd.DataFrame(rows).pivot_table(index='modelo', columns='corte', values='auc'); t['promedio'] = t.mean(axis=1)
print(t.round(4).to_string())
