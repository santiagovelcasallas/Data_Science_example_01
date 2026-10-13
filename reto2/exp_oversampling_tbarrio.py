"""(A) Bootstrap de positivos (random oversampling, como en las diapositivas) y Tomek + RUS 30 %.
(B) t x barrio: tendencia propia de cada barrio (Poisson mensual antes del corte) + Optuna. Base: cal + t."""
import numpy as np, pandas as pd, lightgbm as lgb, optuna, json
from imblearn.under_sampling import TomekLinks
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import roc_auc_score as auc
import exp_lgb as E
optuna.logging.set_verbosity(optuna.logging.WARNING)
p = E.p; CT = E.CAL + ['t']
CORTES = ['2018-07-01', '2018-08-01', '2018-10-01', '2018-12-01', '2019-01-01', '2019-03-01']
def split(c):
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5)
    return a, p[(p.TW < a) & (p.target >= 0)].copy(), p[(p.TW >= a) & (p.TW < b)].copy()
def lgbm(d, F, params=None):
    return lgb.LGBMClassifier(**(params or E.base), verbose=-1).fit(d[F], d.target)
def tabla(rows, titulo):
    t = pd.DataFrame(rows).pivot_table(index='modelo', columns='corte', values='auc'); t['promedio'] = t.mean(axis=1)
    print(titulo); print(t.round(4).sort_values('promedio', ascending=False).to_string(), flush=True)

# ---------- (A) bootstrap de positivos
rows = []
for c in CORTES:
    a, tr, va = split(c)
    rows.append(dict(corte=c[:7], modelo='cal+t (base)', auc=auc(va.target, lgbm(tr, CT).predict_proba(va[CT])[:, 1])))
    pos, neg = tr[tr.target == 1], tr[tr.target == 0]
    for r in (0.1, 0.2, 0.3, 0.4, 0.5):
        n_pos = int(r * len(neg) / (1 - r))                               # proporcion nueva de positivos = r
        d = pd.concat([neg, pos.sample(n_pos, replace=True, random_state=0)])
        rows.append(dict(corte=c[:7], modelo=f'bootstrap positivos {int(r*100)}%', auc=auc(va.target, lgbm(d, CT).predict_proba(va[CT])[:, 1])))
    # Tomek + RUS 30 %: quita negativos en enlaces Tomek y luego submuestrea negativos hasta 30 % de positivos
    X = tr[CT].assign(barrio=tr.barrio.cat.codes)
    tl = TomekLinks(sampling_strategy='majority', n_jobs=4); tl.fit_resample(X, tr.target); keep = tl.sample_indices_
    tk = tr.iloc[keep]; ng = tk[tk.target == 0]; ps = tk[tk.target == 1]
    d = pd.concat([ps, ng.sample(min(len(ng), int(len(ps) * 0.7 / 0.3)), random_state=0)])
    rows.append(dict(corte=c[:7], modelo='tomek + RUS 30%', auc=auc(va.target, lgbm(d, CT).predict_proba(va[CT])[:, 1])))
    print(c, 'A listo', flush=True)
tabla(rows, 'Tabla A. Bootstrap de positivos y Tomek + RUS')

# ---------- (B) t x barrio
def tendencia(tr, a):
    """Pendiente log-lineal por barrio de los conteos mensuales antes del corte (Poisson con L2 suave)."""
    m = tr.groupby(['BARRIO', tr.TW.dt.to_period('M')]).n_acc.sum().reset_index()
    m['tm'] = (m.TW.dt.to_timestamp() - a).dt.days / 365.0
    return {b: PoissonRegressor(alpha=1e-3).fit(g[['tm']], g.n_acc).coef_[0] for b, g in m.groupby('BARRIO')}
DATA = {}
for c in CORTES:
    a, tr, va = split(c)
    sl = tendencia(tr, a)
    for d in (tr, va):
        d['pend_barrio'] = d.BARRIO.map(sl)
        d['nivel_tend'] = d.pend_barrio * (d.TW - a).dt.days / 365.0        # log-nivel extrapolado respecto al corte
    DATA[c] = (tr, va)
CTB = CT + ['pend_barrio', 'nivel_tend']
def score(params, c, F):
    tr, va = DATA[c]; return auc(va.target, lgbm(tr, F, params).predict_proba(va[F])[:, 1])
AJ = ['2018-07-01', '2018-10-01']
def obj(t):
    pr = dict(num_leaves=t.suggest_int('num_leaves', 4, 63, log=True), min_child_samples=t.suggest_int('min_child_samples', 50, 3000, log=True),
              learning_rate=t.suggest_float('learning_rate', 0.005, 0.1, log=True), n_estimators=t.suggest_int('n_estimators', 100, 1500, log=True),
              subsample=t.suggest_float('subsample', 0.5, 1.0), subsample_freq=1, colsample_bytree=t.suggest_float('colsample_bytree', 0.3, 1.0),
              reg_lambda=t.suggest_float('reg_lambda', 1e-3, 100, log=True), min_split_gain=t.suggest_float('min_split_gain', 0, 0.5))
    return np.mean([score(pr, c, CTB) for c in AJ])
st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
st.enqueue_trial({k: v for k, v in E.base.items() if k != 'subsample_freq'} | {'min_split_gain': 0.0})
st.optimize(obj, n_trials=30)
best = st.best_params | {'subsample_freq': 1}; json.dump(best, open('optuna_tbarrio.json', 'w'), indent=1)
print('Optuna: base en ajuste =', round(st.trials[0].value, 4), ' mejor en ajuste =', round(st.best_value, 4), best, flush=True)
rows = []
for c in CORTES:
    for n, F, pr in [('cal+t', CT, None), ('cal+t+t_x_barrio', CTB, None), ('cal+t+t_x_barrio + optuna', CTB, best)]:
        rows.append(dict(corte=c[:7], modelo=n, auc=score(pr, c, F)))
tabla(rows, 'Tabla B. t x barrio y Optuna (2018-07 y 2018-10 = ajuste; 2019-03 = prueba limpia)')
