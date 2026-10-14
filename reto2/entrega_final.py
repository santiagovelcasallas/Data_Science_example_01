"""Entrega final del Reto 2. Una sola funcion entrena con todo lo anterior al corte y predice el periodo siguiente.
La misma funcion se usa (1) para validar en ago-dic 2018 y (2) para generar la entrega de ago-dic 2019.
Modelo: LightGBM global, barrio + calendario + tendencia t, parametros base. Uso: python entrega_final.py"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from features import calendario

PARAMS = dict(n_estimators=400, learning_rate=0.02, num_leaves=15, min_child_samples=400, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5, verbose=-1)   # semilla por defecto de LightGBM (la de todos los experimentos)
VARIABLES = ['barrio', 'h', 'dow', 'tipo_dia', 'mes', 'doy', 'festivo', 'vispera', 'lunes_fest', 'quincena', 'dic_ene', 'feria', 't']

p = pd.read_pickle('data/panel.pkl')                 # rejilla completa barrio x hora 2017-2019 (prep.py); target = -1 sin etiqueta
p = pd.concat([p, calendario(p)], axis=1)
p['barrio'] = p.BARRIO.astype('category')

def entrenar_y_predecir(corte, fin):
    """Entrena con todas las filas etiquetadas < corte y predice [corte, fin]."""
    tr = p[(p.TW < corte) & (p.target >= 0)]
    te = p[(p.TW >= corte) & (p.TW <= fin)].copy()
    modelo = lgb.LGBMClassifier(**PARAMS).fit(tr[VARIABLES], tr.target)
    te['prob'] = modelo.predict_proba(te[VARIABLES])[:, 1]
    return tr, te

# 1. Validacion walk-forward con exactamente el mismo codigo: 6 cortes, cada uno predice los 5 meses siguientes
res = []
for c in ['2018-07-01', '2018-08-01', '2018-10-01', '2018-12-01', '2019-01-01', '2019-03-01']:
    a = pd.Timestamp(c); fin = a + pd.DateOffset(months=5) - pd.Timedelta(hours=1)
    tr, va = entrenar_y_predecir(a, fin)
    assert tr.TW.max() < va.TW.min() and len(va) == 22 * va.TW.nunique() and (va.target >= 0).all()
    res.append(roc_auc_score(va.target, va.prob))
    print(f'VALIDACION {a:%Y-%m} a {fin:%Y-%m}: entrena hasta {tr.TW.max()}, evalua {len(va):,} filas '
          f'({va.target.sum():,} positivos) -> ROC-AUC = {res[-1]:.4f}')
print(f'PROMEDIO 6 cortes = {np.mean(res):.4f}   (min {min(res):.4f}, max {max(res):.4f})')

# 2. Entrega: entrena con todo hasta 2019-07-31 y predice 2019-08-01 00:00 a 2019-12-30 23:00
tr, te = entrenar_y_predecir(pd.Timestamp('2019-08-01'), pd.Timestamp('2019-12-30 23:00'))
assert tr.TW.max() == pd.Timestamp('2019-07-31 23:00') and (te.target < 0).all()
sub = pd.DataFrame({'id': te.BARRIO + '|' + te.TW.dt.strftime('%Y-%m-%d %H:%M:%S'), 'target': te.prob.values})
assert len(sub) == 80256 and sub.id.is_unique and sub.target.between(0, 1).all() and sub.target.notna().all()
sub.to_csv('mi_prediccion.csv', index=False)
print(f'ENTREGA: entrena {len(tr):,} filas hasta {tr.TW.max()}, predice {len(sub):,} filas '
      f'({te.TW.min()} a {te.TW.max()}) -> mi_prediccion.csv')
print(sub.head(3).to_string(index=False)); print(sub.target.describe().round(4).to_string())
