"""Construye el panel barrio x hora (rejilla completa) con target y n_acc."""
import sqlite3, pandas as pd, numpy as np
c = sqlite3.connect('data/data.sqlite3')
cl = pd.read_sql("select * from clima", c, parse_dates=['TW'])
acc = pd.read_sql("select TW, BARRIO from accidentes", c, parse_dates=['TW'])
raw = pd.read_sql("select * from raw_accidentes", c, parse_dates=['TW'])
barrios = sorted(cl.BARRIO.unique())
tw = pd.date_range('2017-01-01', '2019-12-31 23:00', freq='h')
grid = pd.MultiIndex.from_product([barrios, tw], names=['BARRIO', 'TW']).to_frame(index=False)
p = grid.merge(cl, on=['BARRIO', 'TW'], how='left')
acc['target'] = 1
p = p.merge(acc.drop_duplicates(), on=['BARRIO', 'TW'], how='left')
p['target'] = p.target.fillna(0).astype('int8')
n = raw.groupby(['BARRIO', 'TW']).size().rename('n_acc').reset_index()
p = p.merge(n, on=['BARRIO', 'TW'], how='left'); p['n_acc'] = p.n_acc.fillna(0).astype('int16')
p.loc[p.TW > '2019-07-31 23:00', ['target', 'n_acc']] = -1
p.to_pickle('data/panel.pkl'); raw.to_pickle('data/raw.pkl')
tr = p[p.target >= 0]
print(p.shape, tr.shape, tr.target.mean(), (p.target < 0).sum())
