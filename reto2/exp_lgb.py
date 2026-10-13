"""LightGBM global con validacion walk-forward (folds A/B + rodantes)."""
import pandas as pd, numpy as np, lightgbm as lgb, sys
from sklearn.metrics import roc_auc_score as auc
from features import calendario
p = pd.read_pickle('data/panel.pkl')
p = pd.concat([p, calendario(p)], axis=1)
p['barrio'] = p.BARRIO.astype('category')
CLIMA = ['precipIntensity', 'temperature', 'humidity', 'windSpeed', 'cloudCover', 'visibility']
FOLDS = {'A': ('2018-08-01', '2019-01-01'), 'B': ('2019-03-01', '2019-08-01'),
         'R1': ('2018-01-01', '2018-06-01'), 'R2': ('2018-06-01', '2018-11-01'), 'R3': ('2018-11-01', '2019-04-01')}

def te(tr, va, keys, m=200):
    """TE jerarquico: prior = tasa barrio x hora."""
    prior = tr.groupby(['BARRIO', 'h']).target.mean().rename('prior')
    g = tr.groupby(keys).target.agg(['sum', 'count'])
    def apply(df):
        x = df[keys].merge(g, left_on=keys, right_index=True, how='left').fillna(0)
        pr = df.set_index(['BARRIO', 'h']).index.map(prior).values
        return ((x['sum'].values + m * pr) / (x['count'].values + m))
    return apply(tr), apply(va), pr_vals(tr, va, prior)

def pr_vals(tr, va, prior):
    return (tr.set_index(['BARRIO', 'h']).index.map(prior).values, va.set_index(['BARRIO', 'h']).index.map(prior).values)

def run(feats, use_te, params, folds=FOLDS):
    res = {}
    for f, (a, b) in folds.items():
        tr = p[(p.TW < a) & (p.target >= 0)].copy(); va = p[(p.TW >= a) & (p.TW < b)].copy()
        F = list(feats)
        if use_te:
            for keys in [['BARRIO', 'h', 'tipo_dia'], ['BARRIO', 'tipo_dia'], ['BARRIO', 'mes'], ['h', 'tipo_dia']]:
                n = 'te_' + '_'.join(keys); t1, t2, (p1, p2) = te(tr, va, keys)
                tr[n], va[n] = t1, t2; F.append(n)
            tr['te_bh'], va['te_bh'] = p1, p2; F.append('te_bh')
        m = lgb.LGBMClassifier(**params, verbose=-1)
        m.fit(tr[F], tr.target)
        res[f] = auc(va.target, m.predict_proba(va[F])[:, 1])
    return res

CAL = ['barrio', 'h', 'dow', 'tipo_dia', 'mes', 'doy', 'festivo', 'vispera', 'lunes_fest', 'quincena', 'dic_ene', 'feria']
base = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=400, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=5)
if __name__ == "__main__":
  for name, feats, ute in [('cal', CAL, False), ('cal+te', CAL, True), ('cal+te+t', CAL + ['t'], True),
                         ('cal+te+clima', CAL + CLIMA, True)]:
      r = run(feats, ute, base)
      print(name, {k: round(v, 4) for k, v in r.items()}, 'AB=', round((r['A'] + r['B']) / 2, 4), 'all=', round(np.mean(list(r.values())), 4), flush=True)
