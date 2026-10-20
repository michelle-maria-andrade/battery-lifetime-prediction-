"""Train the final model on all eight batteries and save the pipeline and model card.

Run from the project root:  python src/train_final.py
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as ft

root = '.' if os.path.isdir('data') else '..'
model_dir = os.path.join(root, 'models')
os.makedirs(model_dir, exist_ok=True)

df, checks = ft.load_all(root)
print('Data check')
print(checks.to_string(index=False))

cols = ft.features
batteries = sorted(df['battery'].unique())
x_all, y_all, g_all = df[cols].values, df['soh'].values, df['battery'].values

# regularisation strength by leave-one-battery-out over all batteries
scores = {}
for c in [1.0, 10.0, 100.0]:
    scores[c] = np.mean([ft.rmse_of(y_all[g_all == b],
                                    ft.svr_fit(x_all[g_all != b], y_all[g_all != b], c).predict(x_all[g_all == b]))
                         for b in batteries])
best_c = min(scores, key=scores.get)
print('\nMean RMSE by C:', {k: round(v, 4) for k, v in scores.items()}, '| chosen C =', best_c)


rows = []
# nested leave-one-battery-out: for battery b, absolute errors of the other batteries of the same
# dataset, each predicted by a model that has seen neither b nor that battery
rows = []
for b in batteries:
    rest = [k for k in batteries if k != b]
    resid = []
    for o in rest:
        if ft.ds_of[o] != ft.ds_of[b]:
            continue
        m_in = np.isin(g_all, [k for k in rest if k != o])
        resid.append(np.abs(y_all[g_all == o] - ft.svr_fit(x_all[m_in], y_all[m_in], best_c).predict(x_all[g_all == o])))
    q = float(np.quantile(np.concatenate(resid), ft.level))
    mb = g_all == b
    mu = ft.svr_fit(x_all[~mb], y_all[~mb], best_c).predict(x_all[mb])
    err = np.abs(y_all[mb] - mu)
    rows.append({'battery': b, 'dataset': ft.ds_of[b], 'rmse': ft.rmse_of(y_all[mb], mu),
                 'half_width': q, 'coverage': float(np.mean(err <= q))})
cal = pd.DataFrame(rows)
print('\nCalibrated 95 percent intervals (nested leave-one-battery-out)')
print(cal.round(3).to_string(index=False))
by_ds = cal.groupby('dataset')[['rmse', 'half_width', 'coverage']].mean()
print('\nBy dataset')
print(by_ds.round(4).to_string())

# final model on every battery
final = ft.svr_fit(x_all, y_all, best_c)
joblib.dump(final, os.path.join(model_dir, 'final_pipeline.joblib'))

card = {
    'version': '1.0',
    'model': 'SVR (RBF), C=%s, epsilon=0.02, standardised inputs' % best_c,
    'features': cols,
    'is_calce': '1 for CALCE-type cells, 0 for NASA-type cells',
    'feature_definitions': {
        'cc_ratio': 'time to first reach %.2f V during charge / early-life median' % ft.v_cc,
        'cv_ratio': 'time from %.2f V to end of charge / early-life median' % ft.v_cc,
        't40_ratio': 'time to reach %.1f V during charge / early-life median' % ft.v_mid,
        'win_ratio': 'capacity discharged between %.1f V and %.1f V / early-life median' % (ft.v_hi, ft.v_lo)},
    'early_life_cycles': ft.n_early,
    'end_of_life_soh': ft.eol,
    'interval_level': ft.level,
    'interval_half_width_soh': by_ds['half_width'].round(4).to_dict(),
    'loo_rmse_by_dataset': by_ds['rmse'].round(4).to_dict(),
    'nested_coverage_by_dataset': by_ds['coverage'].round(3).to_dict(),
    'training_batteries': batteries,
    'sklearn_version': sklearn.__version__,
}
with open(os.path.join(model_dir, 'model_card.json'), 'w') as fh:
    json.dump(card, fh, indent=2)

reload = joblib.load(os.path.join(model_dir, 'final_pipeline.joblib'))
print('\nSaved to', model_dir, '| reload check, max difference:',
      float(np.max(np.abs(reload.predict(x_all[:5]) - final.predict(x_all[:5])))))
print(json.dumps(card, indent=2))