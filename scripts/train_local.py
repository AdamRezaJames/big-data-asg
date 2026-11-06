"""Fixed, predeclared models. Group CV selects a local model before test evaluation."""
import json
import platform
import time
import importlib.metadata
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from common import ROOT, OUT, NUMERIC, CATEGORICAL, SEED, setup, save_json


def preprocessor():
    return ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), NUMERIC),
        ('district', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ], sparse_threshold=0)


def metrics(y, pred):
    return {'mae_cm': float(mean_absolute_error(y, pred)),
            'rmse_cm': float(np.sqrt(mean_squared_error(y, pred))),
            'r2': float(r2_score(y, pred))}


def export_features(train, val, name):
    prep = preprocessor()
    xtr = prep.fit_transform(train[NUMERIC + CATEGORICAL])
    xva = prep.transform(val[NUMERIC + CATEGORICAL])
    columns = [f'x{i}' for i in range(xtr.shape[1])]
    for data, mat, role in [(train, xtr, 'train'), (val, xva, 'eval')]:
        table = pd.DataFrame(mat, columns=columns)
        table.insert(0, 'record_id', data.record_id.to_numpy())
        table['label'] = data.target_cm.to_numpy()
        table.to_csv(OUT / 'spark_features' / f'{name}_{role}.csv', index=False)
    return list(prep.get_feature_names_out())


def cluster_bootstrap_mae(y, pred, groups):
    rng = np.random.default_rng(SEED)
    keys = np.unique(groups)
    errors = np.abs(y - pred)
    samples = []
    for _ in range(2000):
        selected = rng.choice(keys, len(keys), replace=True)
        sampled = np.concatenate([errors[groups == key] for key in selected])
        samples.append(sampled.mean())
    return [float(x) for x in np.quantile(samples, [.025, .975])]


def main():
    setup()
    (OUT / 'spark_features').mkdir(exist_ok=True)
    (OUT / 'models').mkdir(exist_ok=True)
    df = pd.read_csv(OUT / 'ml_dataset.csv', dtype={'kelurahan_id': str}).fillna({'quality_flags': ''})
    train, test = df[df.split == 'train'].copy(), df[df.split == 'test'].copy()
    feature_names = export_features(train, test, 'final')
    for fold in range(4):
        export_features(train[train.cv_fold != fold], train[train.cv_fold == fold], f'fold{fold}')
    estimators = {
        'Baseline median': DummyRegressor(strategy='median'),
        'Baseline mean': DummyRegressor(strategy='mean'),
        'Ridge': Pipeline([('scale', StandardScaler()), ('regressor', Ridge(alpha=10.0))]),
        'Random Forest sklearn': RandomForestRegressor(n_estimators=200, max_depth=4,
                min_samples_leaf=3, max_features=1.0, bootstrap=True, random_state=SEED, n_jobs=2),
    }
    cv_rows, cv_pred_rows = [], []
    for name, estimator in estimators.items():
        for fold in range(4):
            tr, va = train[train.cv_fold != fold], train[train.cv_fold == fold]
            model = Pipeline([('preprocess', preprocessor()), ('model', clone(estimator))])
            start = time.perf_counter()
            model.fit(tr[NUMERIC + CATEGORICAL], tr.target_cm)
            pred = model.predict(va[NUMERIC + CATEGORICAL])
            elapsed = time.perf_counter() - start
            cv_rows.append({'model': name, 'fold': fold, **metrics(va.target_cm, pred), 'fit_predict_seconds': elapsed})
            cv_pred_rows.extend({'model': name, 'record_id': rid, 'fold': fold, 'actual_cm': float(y), 'prediction_cm': float(p)}
                                for rid, y, p in zip(va.record_id, va.target_cm, pred))
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUT / 'cv_metrics_local.csv', index=False)
    pd.DataFrame(cv_pred_rows).to_csv(OUT / 'cv_predictions_local.csv', index=False)
    ranking = cv.groupby('model').mae_cm.mean().sort_values()
    chosen = str(ranking.index[0])
    rows, predictions, fitted = [], [], {}
    for name, estimator in estimators.items():
        model = Pipeline([('preprocess', preprocessor()), ('model', clone(estimator))])
        start = time.perf_counter()
        model.fit(train[NUMERIC + CATEGORICAL], train.target_cm)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pred = model.predict(test[NUMERIC + CATEGORICAL])
        predict_seconds = time.perf_counter() - start
        scores = metrics(test.target_cm, pred)
        interval = cluster_bootstrap_mae(test.target_cm.to_numpy(), pred, test.kelurahan_id.to_numpy())
        rows.append({'model': name, 'library': 'scikit-learn', **scores,
                     'cv_mae_mean_cm': float(cv[cv.model == name].mae_cm.mean()),
                     'cv_mae_sd_cm': float(cv[cv.model == name].mae_cm.std(ddof=1)),
                     'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds,
                     'mae_ci_low_cm': interval[0], 'mae_ci_high_cm': interval[1],
                     'selected_by_local_cv': name == chosen})
        predictions.extend({'model': name, 'record_id': rid, 'kelurahan_id': k, 'actual_cm': float(y),
                            'prediction_cm': float(p), 'residual_cm': float(y-p)}
                           for rid, k, y, p in zip(test.record_id, test.kelurahan_id, test.target_cm, pred))
        fitted[name] = model
        joblib.dump(model, OUT / 'models' / (name.lower().replace(' ', '_') + '.joblib'))
    pd.DataFrame(rows).to_csv(OUT / 'test_metrics_local.csv', index=False)
    pd.DataFrame(predictions).to_csv(OUT / 'test_predictions_local.csv', index=False)
    # Diagnostic on held-out data; never used for selecting features or retraining.
    model = fitted['Random Forest sklearn']
    base = mean_absolute_error(test.target_cm, model.predict(test[NUMERIC + CATEGORICAL]))
    rng = np.random.default_rng(SEED)
    importance = []
    for label, cols in [('Bulan siklik', ['month_sin', 'month_cos']),
                        ('Lokasi representatif', ['longitude', 'latitude']), ('Kecamatan', ['kecamatan'])]:
        deltas = []
        for _ in range(30):
            altered = test[NUMERIC + CATEGORICAL].copy()
            order = rng.permutation(len(altered))
            altered.loc[:, cols] = altered[cols].to_numpy()[order]
            deltas.append(mean_absolute_error(test.target_cm, model.predict(altered)) - base)
        importance.append({'feature_group': label, 'mae_increase_cm': float(np.mean(deltas)),
                           'permutation_sd_cm': float(np.std(deltas, ddof=1))})
    pd.DataFrame(importance).to_csv(OUT / 'permutation_importance.csv', index=False)
    # Sensitivity only: the single explicit SM->CM assumption is not silently dropped.
    sensitivity = []
    for name in ['Baseline median', 'Ridge', 'Random Forest sklearn']:
        tr = train[~train.quality_flags.str.contains('depth_unit_typo', regex=False)]
        te = test[~test.quality_flags.str.contains('depth_unit_typo', regex=False)]
        m = Pipeline([('preprocess', preprocessor()), ('model', clone(estimators[name]))])
        m.fit(tr[NUMERIC + CATEGORICAL], tr.target_cm)
        sensitivity.append({'model': name, 'train_rows': len(tr), 'test_rows': len(te),
                            **metrics(te.target_cm, m.predict(te[NUMERIC + CATEGORICAL]))})
    pd.DataFrame(sensitivity).to_csv(OUT / 'unit_assumption_sensitivity.csv', index=False)
    save_json(OUT / 'local_run.json', {'executed_at_utc': datetime.now(timezone.utc).isoformat(), 'selected_by_cv': chosen,
        'selection_rule': 'lowest unweighted mean of 4 group-fold MAEs; fixed candidates, no test tuning',
        'feature_names': feature_names, 'feature_count': len(feature_names),
        'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'networkx', 'shapely', 'pyspark']},
        'rf_parameters': estimators['Random Forest sklearn'].get_params(),
        'bootstrap': '2000 resamples of held-out kelurahan, percentile MAE interval; conditional on one fitted model',
        'timing_scope': 'single local run; not a scalability benchmark'})
    print(pd.DataFrame(rows)[['model', 'cv_mae_mean_cm', 'mae_cm', 'rmse_cm', 'r2']].to_string(index=False))


if __name__ == '__main__':
    main()
