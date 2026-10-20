import os
import glob
import numpy as np
import pandas as pd
import scipy.io
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

nasa_batteries = ['B0005', 'B0006', 'B0007', 'B0018']
calce_batteries = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
ds_of = {b: 'NASA' for b in nasa_batteries}
ds_of.update({b: 'CALCE' for b in calce_batteries})

t_col, v_col, i_col = 'Test_Time(s)', 'Voltage(V)', 'Current(A)'
calce_charge, calce_dis = [2, 4], 7

v_mid, v_cc = 4.0, 4.19
v_hi, v_lo = 3.8, 3.4
n_early = 5
eol = 0.80
win, tol = 11, 0.15
level = 0.95

features = ['cc_ratio', 'cv_ratio', 't40_ratio', 'win_ratio', 'is_calce']
core = ['capacity', 'cc_time', 'cv_time', 't40', 'win_cap', 'dis_time']
ratio_src = {'cc_ratio': 'cc_time', 'cv_ratio': 'cv_time', 't40_ratio': 't40',
             'win_ratio': 'win_cap', 'dis_ratio': 'dis_time'}


def find_dirs(root='.'):
    data_dir = os.path.join(root, 'data')
    nasa = os.path.dirname(glob.glob(os.path.join(data_dir, '**', 'B0005.mat'), recursive=True)[0])
    calce = os.path.dirname(glob.glob(os.path.join(data_dir, '**', 'CS2_35'), recursive=True)[0])
    return nasa, calce


def arr(x):
    return np.asarray(x, dtype=float).ravel()


def charge_feats(t, v):
    t, v = arr(t), arr(v)
    out = {'cc_time': np.nan, 'cv_time': np.nan, 't40': np.nan}
    if len(t) < 5:
        return out
    i_mid = np.where(v >= v_mid)[0]
    i_cc = np.where(v >= v_cc)[0]
    if len(i_mid):
        out['t40'] = t[i_mid[0]] - t[0]
    if len(i_cc):
        out['cc_time'] = t[i_cc[0]] - t[0]
        out['cv_time'] = t[-1] - t[i_cc[0]]
    return out


def dis_feats(t, v, i):
    t, v, i = arr(t), arr(v), arr(i)
    out = {'cap': np.nan, 'dis_time': np.nan, 'win_cap': np.nan}
    if len(t) < 5:
        return out
    dq = np.abs(i[1:]) * np.diff(t) / 3600.0
    out['cap'] = float(dq.sum())
    out['dis_time'] = float(t[-1] - t[0])
    j_hi = np.where(v <= v_hi)[0]
    j_lo = np.where(v <= v_lo)[0]
    if len(j_hi) and len(j_lo) and j_lo[0] > j_hi[0]:
        out['win_cap'] = float(dq[j_hi[0]:j_lo[0]].sum())
    return out


def load_nasa(name, nasa_dir):
    mat = scipy.io.loadmat(os.path.join(nasa_dir, name + '.mat'))
    cycles = mat[name][0, 0]['cycle'][0]
    rows, charge, n_dis = [], None, 0
    for c in cycles:
        kind = str(c['type'][0])
        d = c['data'][0, 0]
        if kind == 'charge':
            charge = charge_feats(d['Time'], d['Voltage_measured'])
        elif kind == 'discharge':
            n_dis += 1
            if charge is None:
                continue
            dis = dis_feats(d['Time'], d['Voltage_measured'], d['Current_measured'])
            rows.append({'battery': name, 'dataset': 'NASA', 'cycle_no': n_dis,
                         'capacity': float(arr(d['Capacity'])[0]), **charge,
                         'dis_time': dis['dis_time'], 'win_cap': dis['win_cap']})
    return pd.DataFrame(rows)


def load_calce(name, calce_dir):
    frames = []
    for f in glob.glob(os.path.join(calce_dir, name, '*.xlsx')):
        sheet = next(s for s in pd.read_excel(f, sheet_name=None).values()
                     if 'Cycle_Index' in s.columns and 'Step_Index' in s.columns)
        frames.append((pd.to_datetime(sheet['Date_Time'].iloc[0], errors='coerce'), sheet))
    frames.sort(key=lambda p: p[0])
    rows, n_cycle = [], 0
    for _, sheet in frames:
        for _, sub in sheet.groupby('Cycle_Index', sort=True):
            ch = sub[sub['Step_Index'].isin(calce_charge)].sort_values(t_col)
            di = sub[sub['Step_Index'] == calce_dis].sort_values(t_col)
            if len(ch) < 10 or len(di) < 10:
                continue
            dis = dis_feats(di[t_col].values, di[v_col].values, di[i_col].values)
            n_cycle += 1
            rows.append({'battery': name, 'dataset': 'CALCE', 'cycle_no': n_cycle,
                         'capacity': dis['cap'], **charge_feats(ch[t_col].values, ch[v_col].values),
                         'dis_time': dis['dis_time'], 'win_cap': dis['win_cap']})
    return pd.DataFrame(rows)


def add_ratios(g):
    ref = g.head(n_early)[core].median()
    g = g.copy()
    g['soh'] = g['capacity'] / ref['capacity']
    for col, src in ratio_src.items():
        g[col] = g[src] / ref[src]
    return g


def prep(g):
    name = g['battery'].iloc[0]
    n0 = len(g)
    g = g.sort_values('cycle_no').dropna(subset=core)
    g = g[(g[core] > 0).all(axis=1)].reset_index(drop=True)
    n_bad = n0 - len(g)
    g = add_ratios(g)
    roll = g['soh'].rolling(win, center=True, min_periods=1).median()
    keep = ((g['soh'] - roll).abs() <= tol) & g['soh'].between(0.2, 1.5)
    n_out = int((~keep).sum())
    g = g[keep].reset_index(drop=True)
    smooth = g['soh'].rolling(5, min_periods=1).median()
    eol_cycle = float(g['cycle_no'].iloc[np.where(smooth.values <= eol)[0][0]])
    g['eol_cycle'] = eol_cycle
    g['rul_cycles'] = np.where(g['cycle_no'] <= eol_cycle, eol_cycle - g['cycle_no'], np.nan)
    g['rul_frac'] = g['rul_cycles'] / eol_cycle
    info = {'battery': name, 'dataset': ds_of[name], 'cycles_loaded': n0,
            'dropped_missing_or_nonpositive': n_bad, 'dropped_outliers': n_out,
            'cycles_kept': len(g), 'final_SoH': round(float(g['soh'].iloc[-1]), 3),
            'min_SoH': round(float(g['soh'].min()), 3), 'eol_cycle': eol_cycle}
    return g, info


def load_all(root='.'):
    nasa_dir, calce_dir = find_dirs(root)
    raw = pd.concat([load_nasa(b, nasa_dir) for b in nasa_batteries] +
                    [load_calce(b, calce_dir) for b in calce_batteries], ignore_index=True)
    cleaned, infos = zip(*[prep(g) for _, g in raw.groupby('battery', sort=False)])
    df = pd.concat(cleaned, ignore_index=True)
    df['is_calce'] = (df['dataset'] == 'CALCE').astype(int)
    return df, pd.DataFrame(infos)


def svr_fit(x, y, c):
    pipe = Pipeline([('sc', StandardScaler()), ('est', SVR(kernel='rbf', C=c, epsilon=0.02))])
    return pipe.fit(x, y)


def rmse_of(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def predict_with_band(pipe, card, frame, is_calce):
    x = frame[features[:-1]].copy()
    x['is_calce'] = is_calce
    mu = pipe.predict(x[features].values)
    half = card['interval_half_width_soh']['CALCE' if is_calce else 'NASA']
    return mu, mu - half, mu + half


def first_crossing(cycles, soh_pred, threshold=eol):
    """Cycle at which the smoothed predicted SoH first reaches the threshold (detection, not forecasting)."""
    smooth = pd.Series(soh_pred).rolling(5, min_periods=1).median().values
    hit = np.where(smooth <= threshold)[0]
    return float(cycles[hit[0]]) if len(hit) else np.nan
