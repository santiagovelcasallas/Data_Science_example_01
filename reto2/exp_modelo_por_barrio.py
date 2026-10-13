"""Cual modelo se adapta mejor a cada barrio. Se elige por barrio con log-loss en folds anteriores y se aplica en un fold de prueba."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score as auc, log_loss
import exp_lgb as E
p = E.p
for a_, b_ in [('BARRIO', 'h'), ('BARRIO', 'tipo_dia'), ('h', 'tipo_dia')]:
    p[f'{a_}_{b_}'] = p[a_].astype(str) + '_' + p[b_].astype(str)
p['h_tipo'] = p.h.astype(str) + '_' + p.tipo_dia.astype(str)
CT = E.CAL + ['t']; CT_LOC = [c for c in CT if c != 'barrio']
LOG_G = ['BARRIO', 'h', 'tipo_dia', 'mes', 'festivo', 'vispera', 'lunes_fest', 'quincena', 'dic_ene', 'feria', 'BARRIO_h', 'BARRIO_tipo_dia', 'h_tipo_dia']
LOG_L = ['h', 'tipo_dia', 'h_tipo', 'mes', 'festivo', 'vispera', 'lunes_fest', 'quincena', 'dic_ene', 'feria']
LOC = dict(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5)
MODELOS = ['LGB global', 'logistica global', 'tasa historica', 'LGB local', 'logistica local']

def logit(tr, va, cols, C):
    enc = OneHotEncoder(handle_unknown='ignore').fit(tr[cols].astype(str))
    m = LogisticRegression(C=C, max_iter=2000).fit(enc.transform(tr[cols].astype(str)), tr.target)
    return m.predict_proba(enc.transform(va[cols].astype(str)))[:, 1]

def predicciones(c):
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5)
    tr = p[(p.TW < a) & (p.target >= 0)]; va = p[(p.TW >= a) & (p.TW < b)].copy()
    va['LGB global'] = lgb.LGBMClassifier(**E.base, verbose=-1).fit(tr[CT], tr.target).predict_proba(va[CT])[:, 1]
    va['logistica global'] = logit(tr, va, LOG_G, 0.03)
    # tasa historica barrio x hora x tipo_dia, suavizada hacia barrio x hora
    bh = tr.groupby(['BARRIO', 'h']).target.mean(); st = tr.groupby(['BARRIO', 'h', 'tipo_dia']).target.agg(['sum', 'count'])
    k = pd.MultiIndex.from_frame(va[['BARRIO', 'h', 'tipo_dia']])
    pr = bh.reindex(pd.MultiIndex.from_frame(va[['BARRIO', 'h']])).values
    va['tasa historica'] = (st['sum'].reindex(k).fillna(0).values + 100 * pr) / (st['count'].reindex(k).fillna(0).values + 100)
    va['LGB local'] = np.nan; va['logistica local'] = np.nan
    for bar in va.BARRIO.unique():
        t_, v_ = tr[tr.BARRIO == bar], va.BARRIO == bar
        va.loc[v_, 'LGB local'] = lgb.LGBMClassifier(**LOC, verbose=-1).fit(t_[CT_LOC], t_.target).predict_proba(va.loc[v_, CT_LOC])[:, 1]
        va.loc[v_, 'logistica local'] = logit(t_, va[v_], LOG_L, 0.1)
    print(c, 'listo', flush=True)
    return va[['BARRIO', 'target'] + MODELOS]

def ll_por_barrio(va):
    return pd.DataFrame({m: va.groupby('BARRIO').apply(lambda g: log_loss(g.target, g[m].clip(1e-6, 1 - 1e-6), labels=[0, 1])) for m in MODELOS})

ESQ = {'Esquema 1 (elige 2018-01 y 2018-03, prueba 2018-08)': (['2018-01-01', '2018-03-01'], '2018-08-01'),
       'Esquema 2 (elige 2018-07 y 2018-10, prueba 2019-03)': (['2018-07-01', '2018-10-01'], '2019-03-01')}
P = {c: predicciones(c) for c in sorted({x for s, t in ESQ.values() for x in s + [t]})}
eleccion, filas = {}, []
for n, (sel, test) in ESQ.items():
    ll = sum(ll_por_barrio(P[c]) for c in sel)                     # log-loss sumado en los folds de seleccion
    elegido = ll.idxmin(axis=1); eleccion[n] = elegido
    va = P[test]; ll_test = ll_por_barrio(va); oraculo = ll_test.idxmin(axis=1)
    sel_score = np.select([va.BARRIO == b for b in elegido.index], [va[elegido[b]] for b in elegido.index])
    ora_score = np.select([va.BARRIO == b for b in oraculo.index], [va[oraculo[b]] for b in oraculo.index])
    r = {m: auc(va.target, va[m]) for m in MODELOS}
    r['seleccion por barrio'] = auc(va.target, sel_score); r['oraculo (elige con la prueba, NO valido)'] = auc(va.target, ora_score)
    filas.append(pd.Series(r, name=n))
    # AUC dentro de cada barrio en la prueba, para ver diferencias reales entre modelos
    w = pd.DataFrame({m: va.groupby('BARRIO').apply(lambda g: auc(g.target, g[m])) for m in MODELOS})
    w['elegido'] = elegido; w['positivos prueba'] = va.groupby('BARRIO').target.sum()
    print(f'\nTabla por barrio, {n}: AUC dentro del barrio en la prueba y modelo elegido')
    print(w.round(3).to_string(), flush=True)
print('\nTabla resumen. AUC global en el fold de prueba')
print(pd.DataFrame(filas).T.round(4).to_string())
e = pd.DataFrame(eleccion); e.columns = ['esquema 1', 'esquema 2']; e['misma eleccion'] = e['esquema 1'] == e['esquema 2']
print('\nTabla de elecciones por barrio'); print(e.to_string()); print('barrios con la misma eleccion en ambos esquemas:', e['misma eleccion'].sum(), 'de', len(e))
print('\nConteo de elecciones'); print(e[['esquema 1', 'esquema 2']].apply(pd.Series.value_counts).fillna(0).astype(int).to_string())
