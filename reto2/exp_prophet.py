"""LightGBM + Prophet: Prophet por barrio sobre conteos diarios (festivos CO, estacionalidad semanal y anual),
su pronostico diario entra como variable del LightGBM horario. Walk-forward en los mismos 5 folds."""
import logging, numpy as np, pandas as pd, lightgbm as lgb
from prophet import Prophet
from sklearn.metrics import roc_auc_score as auc
import exp_lgb as E
from features import FESTIVOS
logging.getLogger('cmdstanpy').setLevel(logging.ERROR); logging.getLogger('prophet').setLevel(logging.ERROR)
p = E.p
p['d'] = p.TW.dt.normalize()
dia = p[p.target >= 0].groupby(['BARRIO', 'd']).n_acc.sum().reset_index()
hol = pd.DataFrame({'holiday': 'festivo', 'ds': FESTIVOS})

def prophet_feats(a, b):
    """Ajusta con datos < a; devuelve yhat diario por barrio (in-sample para train, pronostico para val)."""
    out = []
    for bar, g in dia[dia.d < a].groupby('BARRIO'):
        m = Prophet(holidays=hol, weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                    changepoint_prior_scale=0.05)
        m.fit(g.rename(columns={'d': 'ds', 'n_acc': 'y'})[['ds', 'y']])
        fut = pd.DataFrame({'ds': pd.date_range('2017-01-01', pd.Timestamp(b) - pd.Timedelta(days=1))})
        fc = m.predict(fut)[['ds', 'yhat', 'trend', 'weekly', 'yearly']]
        fc['BARRIO'] = bar; out.append(fc)
    f = pd.concat(out).rename(columns={'ds': 'd', 'yhat': 'pr_yhat', 'trend': 'pr_trend', 'weekly': 'pr_week', 'yearly': 'pr_year'})
    return f

PR = ['pr_yhat', 'pr_trend', 'pr_week', 'pr_year']
CAL = E.CAL
res = {k: {} for k in ['lgb', 'lgb+prophet', 'prophet_solo', 'lgb*prophet']}
for f, (a, b) in E.FOLDS.items():
    pf = prophet_feats(a, b)
    q = p[(p.TW < b) & (p.target >= 0)].merge(pf, on=['BARRIO', 'd'], how='left')
    tr, va = q[q.TW < a], q[q.TW >= a]
    m0 = lgb.LGBMClassifier(**E.base, verbose=-1).fit(tr[CAL], tr.target); s0 = m0.predict_proba(va[CAL])[:, 1]
    m1 = lgb.LGBMClassifier(**E.base, verbose=-1).fit(tr[CAL + PR], tr.target); s1 = m1.predict_proba(va[CAL + PR])[:, 1]
    # prophet solo: nivel diario x perfil horario del barrio
    prof = tr.groupby(['BARRIO', 'h']).target.mean() / tr.groupby('BARRIO').target.mean()
    sp = va.pr_yhat.clip(lower=1e-3).values * va.set_index(['BARRIO', 'h']).index.map(prof).values
    res['lgb'][f] = auc(va.target, s0); res['lgb+prophet'][f] = auc(va.target, s1)
    res['prophet_solo'][f] = auc(va.target, sp)
    res['lgb*prophet'][f] = auc(va.target, s0 * va.pr_yhat.clip(lower=1e-3).values / va.groupby('BARRIO').pr_yhat.transform('mean').values)
    print(f, {k: round(v[f], 4) for k, v in res.items()}, flush=True)
print('PROMEDIO', {k: (round((v['A'] + v['B']) / 2, 4), round(np.mean(list(v.values())), 4)) for k, v in res.items()}, '(A/B, 5 folds)')
