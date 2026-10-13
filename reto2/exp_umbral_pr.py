"""Curva PR, umbral por costo y por capacidad de patrullas. Modelo: LightGBM barrio + calendario.
Umbral elegido en el fold 2018-08 (ago-dic 2018) y aplicado sin cambios al fold 2019-03 (mar-jul 2019)."""
import numpy as np, pandas as pd, lightgbm as lgb, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, confusion_matrix
import exp_lgb as E
p = E.p
AZUL, NARANJA, GRIS, TXT, SUP = '#2a78d6', '#eb6834', '#8a8984', '#52514e', '#fcfcfb'
plt.rcParams.update({'figure.facecolor': SUP, 'axes.facecolor': SUP, 'axes.edgecolor': '#d8d7d3', 'axes.labelcolor': TXT,
                     'xtick.color': TXT, 'ytick.color': TXT, 'axes.grid': True, 'grid.color': '#ecebe8', 'grid.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
def fold(c):
    a = pd.Timestamp(c); b = a + pd.DateOffset(months=5)
    tr = p[(p.TW < a) & (p.target >= 0)]; va = p[(p.TW >= a) & (p.TW < b)].copy()
    va['prob'] = lgb.LGBMClassifier(**E.base, verbose=-1).fit(tr[E.CAL], tr.target).predict_proba(va[E.CAL])[:, 1]
    return va
F = {'2018-08 (ago-dic 2018)': fold('2018-08-01'), '2019-03 (mar-jul 2019)': fold('2019-03-01')}
SEL, APL = list(F)

# 1. Curva PR
fig, ax = plt.subplots(figsize=(6.4, 4.2))
print('Tabla 1. Metricas globales por fold')
for (n, va), col in zip(F.items(), [AZUL, NARANJA]):
    pr, rc, _ = precision_recall_curve(va.target, va.prob)
    ap = average_precision_score(va.target, va.prob)
    ax.plot(rc, pr, color=col, lw=2, label=f'{n}: PR-AUC {ap:.3f}')
    print(f'  {n}: ROC-AUC={roc_auc_score(va.target, va.prob):.4f}  PR-AUC={ap:.4f}  prevalencia={va.target.mean():.4f}  lift PR-AUC={ap / va.target.mean():.2f}x')
ax.axhline(F[SEL].target.mean(), color=GRIS, lw=1, ls='--', label=f'azar (prevalencia {F[SEL].target.mean():.3f})')
ax.set(xlabel='Recall (accidentes detectados)', ylabel='Precisión (alertas con accidente)', ylim=(0, 0.2),
       title='Figura 1. Curva precisión-recall, LightGBM barrio + calendario')
ax.legend(frameon=False, fontsize=9); fig.tight_layout(); fig.savefig('fig_pr.png', dpi=130)

# 2. Umbral por costo: costo = C_FP * FP + C_FN * FN, con C_FP = 1 (patrulla sin accidente)
def costo(va, t, k):
    tn, fp, fn, tp = confusion_matrix(va.target, va.prob >= t, labels=[0, 1]).ravel()
    return fp + k * fn, tn, fp, fn, tp
grid = np.quantile(F[SEL].prob, np.linspace(0, 0.999, 400))
rows = []
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for k, col in zip([10, 25, 50, 100], [AZUL, NARANJA, '#1baf7a', '#6250d6']):
    cs = np.array([costo(F[SEL], t, k)[0] for t in grid]); base = costo(F[SEL], 1.1, k)[0]   # base = no patrullar nada
    t_opt = grid[cs.argmin()]
    ax.plot(grid, cs / base, color=col, lw=2, label=f'C_FN/C_FP = {k}')
    for n in F:
        c_, tn, fp, fn, tp = costo(F[n], t_opt, k); c0 = costo(F[n], 1.1, k)[0]
        rows.append({'C_FN/C_FP': k, 'umbral (elegido en 2018-08)': round(t_opt, 4), 'umbral teorico 1/(1+k)': round(1 / (1 + k), 4),
                     'fold': n, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'alertas %': round(100 * (tp + fp) / len(F[n]), 1),
                     'recall': round(tp / (tp + fn), 3), 'precision': round(tp / max(tp + fp, 1), 3),
                     'ahorro vs no patrullar %': round(100 * (1 - c_ / c0), 1)})
ax.set(xscale='log', xlabel='Umbral de probabilidad (escala log)', ylabel='Costo relativo a no patrullar',
       title='Figura 2. Costo esperado según umbral (fold 2018-08)')
ax.legend(frameon=False, fontsize=9); fig.tight_layout(); fig.savefig('fig_costo.png', dpi=130)
print('\nTabla 2. Umbral por costo (C_FP = 1: patrulla sin accidente; C_FN = k: accidente no atendido)')
print(pd.DataFrame(rows).to_string(index=False))

# 3. Capacidad: N patrullas x 4 h/dia -> se cubren las N*4 parejas barrio-hora de mayor probabilidad cada dia
rows = []
for n, va in F.items():
    va['d'] = va.TW.dt.normalize(); va['rk'] = va.groupby('d').prob.rank(ascending=False, method='first')
    for N in [1, 2, 3, 5, 10]:
        cub = va.rk <= N * 4
        rows.append({'fold': n, 'patrullas': N, 'horas-barrio cubiertas/dia': N * 4,
                     '% del total': round(100 * N * 4 / (22 * 24), 1), 'accidentes cubiertos %': round(100 * va.target[cub].sum() / va.target.sum(), 1),
                     'lift': round((va.target[cub].mean()) / va.target.mean(), 2)})
print('\nTabla 3. Cobertura con N patrullas x 4 h/dia (top N*4 parejas barrio-hora de cada dia)')
print(pd.DataFrame(rows).to_string(index=False))

fig, ax = plt.subplots(figsize=(6.4, 4.2))
for (n, va), col in zip(F.items(), [AZUL, NARANJA]):
    o = va.sort_values('prob', ascending=False)
    ax.plot(np.arange(1, len(o) + 1) / len(o), o.target.cumsum() / o.target.sum(), color=col, lw=2, label=n)
ax.plot([0, 1], [0, 1], color=GRIS, lw=1, ls='--', label='azar')
ax.set(xlabel='Fracción de parejas barrio-hora vigiladas (de mayor a menor probabilidad)', ylabel='Fracción de accidentes cubiertos',
       title='Figura 3. Ganancia acumulada (cumulative gains)')
ax.legend(frameon=False, fontsize=9); fig.tight_layout(); fig.savefig('fig_gains.png', dpi=130)
