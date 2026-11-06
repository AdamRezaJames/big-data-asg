"""Audit split leakage, predictions, CV coverage and graph comparison invariants."""
import json
import re
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from common import ROOT, OUT, FIG, SEED, save_json, sha256
from train_local import metrics


def main():
    df = pd.read_csv(OUT / 'ml_dataset.csv', dtype={'kelurahan_id': str})
    original = pd.read_csv(ROOT / 'data/silver/banjir_jakarta_timur_2020.csv', dtype={'kelurahan_id': str})
    assert len(df) == len(original) == 153 and df.record_id.is_unique
    source = original.set_index('record_id')
    assert np.array_equal(df.target_cm.to_numpy(), source.loc[df.record_id, 'depth_max_cm'].to_numpy())
    train, test = df[df.split == 'train'], df[df.split == 'test']
    assert set(train.kelurahan_id).isdisjoint(test.kelurahan_id)
    assert set(train.record_id).isdisjoint(test.record_id)
    for fold in range(4):
        assert set(train[train.cv_fold != fold].kelurahan_id).isdisjoint(train[train.cv_fold == fold].kelurahan_id)
    predictions = pd.concat([pd.read_csv(OUT / 'test_predictions_local.csv', dtype={'kelurahan_id': str}),
                             pd.read_csv(OUT / 'test_predictions_spark.csv', dtype={'kelurahan_id': str})], ignore_index=True)
    tables = pd.concat([pd.read_csv(OUT / 'test_metrics_local.csv'), pd.read_csv(OUT / 'test_metrics_spark.csv')], ignore_index=True)
    for name, subset in predictions.groupby('model'):
        assert len(subset) == 40 and subset.record_id.is_unique
        assert set(subset.record_id) == set(test.record_id)
        assert np.isfinite(subset.prediction_cm).all()
        expected = test.set_index('record_id').loc[subset.record_id, 'target_cm']
        assert np.array_equal(subset.actual_cm, expected)
        saved = tables[tables.model == name].iloc[0]
        for key, val in metrics(subset.actual_cm, subset.prediction_cm).items():
            assert np.isclose(saved[key], val, rtol=1e-10, atol=1e-10), (name, key)
        macro = (subset.assign(abs_error=np.abs(subset.actual_cm-subset.prediction_cm))
                 .groupby('kelurahan_id').abs_error.mean().mean())
        tables.loc[tables.model == name, 'macro_kelurahan_mae_cm'] = macro
    tables.to_csv(OUT / 'all_test_metrics.csv', index=False)
    cvs = pd.concat([pd.read_csv(OUT / 'cv_predictions_local.csv'), pd.read_csv(OUT / 'cv_predictions_spark.csv')])
    for name, subset in cvs.groupby('model'):
        assert len(subset) == len(train) and subset.record_id.is_unique
        assert set(subset.record_id) == set(train.record_id)
        assert set(subset.fold) == set(range(4))
    local = json.loads((OUT / 'local_run.json').read_text())
    assert not any('depth' in f or 'target' in f or 'exposure' in f for f in local['feature_names'])
    spark = json.loads((OUT / 'spark_run.json').read_text())
    assert spark['status'] == 'executed' and spark['master'] == 'local[2]'
    assert local['feature_count'] == spark['feature_count']
    # Paired bootstrap differences are more relevant than overlapping marginal intervals.
    pivot = predictions.pivot(index='record_id', columns='model', values='prediction_cm')
    truth = test.set_index('record_id').loc[pivot.index]
    keys = truth.kelurahan_id.unique()
    errors = pivot.sub(truth.target_cm, axis=0).abs()
    paired = []
    for name in ['Random Forest sklearn', 'Random Forest Spark']:
        for baseline in ['Baseline mean', 'Baseline median']:
            delta = errors[name] - errors[baseline]
            rng = np.random.default_rng(SEED)
            boot = []
            for _ in range(2000):
                sample = rng.choice(keys, len(keys), replace=True)
                idx = np.concatenate([np.flatnonzero(truth.kelurahan_id.to_numpy() == k) for k in sample])
                boot.append(delta.iloc[idx].mean())
            lo, hi = np.quantile(boot, [.025, .975])
            paired.append({'model': name, 'baseline': baseline, 'mae_difference_cm': float(delta.mean()),
                           'ci_low_cm': float(lo), 'ci_high_cm': float(hi)})
    pd.DataFrame(paired).to_csv(OUT / 'paired_mae_comparison.csv', index=False)
    routes = pd.read_csv(OUT / 'graph_routes.csv')
    assert len(routes) == 14
    for _, group in routes.groupby('destination'):
        baseline = group[group['lambda'] == 0].iloc[0]
        assert (group.distance_m >= baseline.distance_m - 1e-7).all()
        for r in group.itertuples(index=False):
            lam, u = r[1], r[2]
            assert r.objective_cost_m <= baseline.distance_m + lam*(baseline.exposure_weighted_m + u*baseline.unknown_length_m) + 1e-6
    graph = json.loads((OUT / 'graph_summary.json').read_text())
    assert graph['nodes'] == 146674 and graph['directed_edges'] == 296891
    assert graph['known_exposure_segments'] + graph['unknown_exposure_segments'] == graph['segments']
    report = {'verified_at': datetime.now(timezone(timedelta(hours=7))).isoformat(),
              'rows': len(df), 'test_predictions_verified': len(predictions),
              'cv_predictions_verified': len(cvs), 'models_evaluated': int(predictions.model.nunique()),
              'disjoint_kelurahan_holdout': True, 'disjoint_kelurahan_cv': True,
              'metrics_recomputed_match': True, 'spark_really_executed': True,
              'same_feature_dimension': local['feature_count'], 'graph_route_invariants_passed': True,
              'paired_comparisons': paired}
    save_json(OUT / 'verification.json', report)
    paths = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json']
    save_json(OUT / 'artifact_manifest.json', [{'file': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
        'sha256': sha256(p)} for p in sorted(paths)])
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
