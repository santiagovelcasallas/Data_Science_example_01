"""Verifica que cada fold evalua sobre TODAS las parejas barrio x hora del periodo, sin filtrar, y sin solaparse con el entrenamiento."""
import pandas as pd
p = pd.read_pickle('data/panel.pkl')
CORTES = ['2018-07-01', '2018-08-01', '2018-10-01', '2018-12-01', '2019-01-01', '2019-03-01']
rows = []
for c in CORTES:
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5)
    tr = p[(p.TW < a) & (p.target >= 0)]; va = p[(p.TW >= a) & (p.TW < b)]
    esperadas = 22 * int((b - a) / pd.Timedelta(hours=1))
    rows.append(dict(corte=c[:7], periodo=f'{a:%Y-%m-%d} a {b - pd.Timedelta(hours=1):%Y-%m-%d %H}h', filas_evaluadas=len(va),
                     filas_esperadas=esperadas, completo=len(va) == esperadas, barrios=va.BARRIO.nunique(),
                     horas_distintas=va.TW.nunique(), etiquetas_faltantes=(va.target < 0).sum(), positivos=int(va.target.sum()),
                     filas_sin_clima=int(va.temperature.isna().sum()), ultimo_train=tr.TW.max(), solape=(tr.TW.max() >= a)))
print(pd.DataFrame(rows).to_string(index=False))
print('\nPeriodo del leaderboard (sin etiquetas):', (p.target < 0).sum() - 22 * 24, 'filas de 2019-08-01 a 2019-12-30 (el README dice 80.256)')
