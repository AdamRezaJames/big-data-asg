"""Create one ML row per source flood report, with group-disjoint holdout."""
import json
import numpy as np
import pandas as pd
from shapely.geometry import shape
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from common import ROOT, OUT, SEED, setup, save_json, sha256


def main():
    setup()
    source = ROOT / 'data/silver/banjir_jakarta_timur_2020.csv'
    boundary_path = ROOT / 'data/silver/batas_kelurahan_dki.geojson'
    df = pd.read_csv(source, dtype={'kelurahan_id': str, 'period': str}).fillna({'quality_flags': ''})
    boundaries = json.loads(boundary_path.read_text(encoding='utf-8'))['features']
    points = {}
    for f in boundaries:
        p = shape(f['geometry']).representative_point()
        points[str(f['properties']['kelurahan_id'])] = (p.x, p.y)
    df['longitude'] = df.kelurahan_id.map(lambda k: points[k][0])
    df['latitude'] = df.kelurahan_id.map(lambda k: points[k][1])
    df['month'] = df.period.str[-2:].astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.month / 12)
    df['target_cm'] = pd.to_numeric(df.depth_max_cm, errors='raise')
    assert df.record_id.is_unique and len(df) == 153
    assert df.target_cm.notna().all() and df.target_cm.ge(0).all()
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(df, groups=df.kelurahan_id))
    df['split'] = 'train'
    df.loc[test, 'split'] = 'test'
    df['cv_fold'] = -1
    train_df = df.iloc[train]
    folds = []
    for i, (tr, va) in enumerate(GroupKFold(n_splits=4).split(train_df, groups=train_df.kelurahan_id)):
        ti, vi = train_df.iloc[tr].index, train_df.iloc[va].index
        df.loc[vi, 'cv_fold'] = i
        assert set(df.loc[ti, 'kelurahan_id']).isdisjoint(df.loc[vi, 'kelurahan_id'])
        folds.append({'fold': i, 'train_rows': len(ti), 'validation_rows': len(vi),
                      'train_groups': int(df.loc[ti, 'kelurahan_id'].nunique()),
                      'validation_groups': int(df.loc[vi, 'kelurahan_id'].nunique())})
    assert set(df.loc[train, 'kelurahan_id']).isdisjoint(df.loc[test, 'kelurahan_id'])
    df.to_csv(OUT / 'ml_dataset.csv', index=False)
    df[['record_id', 'kelurahan_id', 'split', 'cv_fold']].to_csv(OUT / 'split_manifest.csv', index=False)
    summary = {
        'seed': SEED, 'rows': len(df), 'groups': int(df.kelurahan_id.nunique()),
        'target': 'upper bound of reported water depth in cm; conditional on an existing report',
        'target_min_cm': float(df.target_cm.min()), 'target_max_cm': float(df.target_cm.max()),
        'target_mean_cm': float(df.target_cm.mean()), 'target_median_cm': float(df.target_cm.median()),
        'months': {str(k): int(v) for k, v in df.month.value_counts().sort_index().items()},
        'splits': {s: {'rows': len(g), 'groups': int(g.kelurahan_id.nunique()),
                       'mean_target_cm': float(g.target_cm.mean()), 'max_target_cm': float(g.target_cm.max())}
                   for s, g in df.groupby('split')},
        'folds': folds,
        'features': ['month_sin', 'month_cos', 'longitude', 'latitude', 'kecamatan (one-hot, fit on train only)'],
        'excluded_predictors': ['depth_raw', 'depth_min_cm', 'depth_max_cm', 'exposure_index',
                                'rw_raw', 'rw_list_json', 'date_end', 'record_id', 'kelurahan_id'],
        'quality_flags': df.quality_flags.value_counts().to_dict(),
        'sources': [{'file': str(p.relative_to(ROOT)), 'sha256': sha256(p)} for p in (source, boundary_path)],
        'evaluation_scope': 'unseen kelurahan, retrospective within 2020; not temporal forecasting',
    }
    save_json(OUT / 'dataset_summary.json', summary)
    print(json.dumps({'rows': len(df), 'splits': summary['splits']}, indent=2))


if __name__ == '__main__':
    main()
