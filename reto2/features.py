"""Variables de calendario (festivos de Colombia 2017-2019) y clima."""
import pandas as pd, numpy as np
FESTIVOS = pd.to_datetime("""2017-01-01 2017-01-09 2017-03-20 2017-04-13 2017-04-14 2017-05-01 2017-05-29 2017-06-19 2017-06-26
2017-07-03 2017-07-20 2017-08-07 2017-08-21 2017-10-16 2017-11-06 2017-11-13 2017-12-08 2017-12-25
2018-01-01 2018-01-08 2018-03-19 2018-03-29 2018-03-30 2018-05-01 2018-05-14 2018-06-04 2018-06-11 2018-07-02
2018-07-20 2018-08-07 2018-08-20 2018-10-15 2018-11-05 2018-11-12 2018-12-08 2018-12-25
2019-01-01 2019-01-07 2019-03-25 2019-04-18 2019-04-19 2019-05-01 2019-06-03 2019-06-24 2019-07-01 2019-07-20
2019-08-07 2019-08-19 2019-10-14 2019-11-04 2019-11-11 2019-12-08 2019-12-25""".split())

def calendario(df):
    ts = df.TW; d = ts.dt.normalize()
    out = pd.DataFrame(index=df.index)
    out['h'] = ts.dt.hour; out['dow'] = ts.dt.dayofweek; out['mes'] = ts.dt.month
    out['doy'] = ts.dt.dayofyear; out['dom'] = ts.dt.day
    fest = d.isin(FESTIVOS)
    out['festivo'] = fest.astype(int)
    out['tipo_dia'] = np.where(fest, 6, out.dow)
    nolab = fest | (out.dow >= 5)
    nl = pd.Series(nolab.values, index=d.values)
    dd = pd.Series(nolab.values, index=df.index)
    out['vispera'] = (d + pd.Timedelta(days=1)).isin(FESTIVOS).astype(int)
    out['lunes_fest'] = (fest & (out.dow == 0)).astype(int)
    out['quincena'] = out.dom.isin([14, 15, 16, 29, 30, 31, 1]).astype(int)
    out['dic_ene'] = (((out.mes == 12) & (out.dom >= 15)) | ((out.mes == 1) & (out.dom <= 10))).astype(int)
    out['feria'] = ((out.mes == 8) & (out.dom <= 12)).astype(int)
    out['t'] = (ts - pd.Timestamp('2017-01-01')).dt.days / 365.0
    out['hs'] = np.sin(2 * np.pi * out.h / 24); out['hc'] = np.cos(2 * np.pi * out.h / 24)
    out['ys'] = np.sin(2 * np.pi * out.doy / 365.25); out['yc'] = np.cos(2 * np.pi * out.doy / 365.25)
    return out
