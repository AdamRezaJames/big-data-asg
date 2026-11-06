"""Train real Spark MLlib RF on the same train-only feature matrices as sklearn.

local[2] is a single-machine Spark run, not a multi-machine scalability test.
No Python UDF is required for fitting or predicting.
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, OUT, SEED, setup, save_json
from train_local import metrics, cluster_bootstrap_mae


def main():
    setup()
    os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import RandomForestRegressor
    print('Starting Spark local[2] with Java; this may take a moment...', flush=True)
    start = time.perf_counter()
    spark = (SparkSession.builder.master('local[2]').appName('Assignment-II-Flood-Regression')
             .config('spark.ui.enabled', 'false')
             .config('spark.sql.shuffle.partitions', '2')
             .config('spark.driver.bindAddress', '127.0.0.1')
             .config('spark.driver.host', '127.0.0.1')
             .config('spark.sql.warehouse.dir', (OUT / 'spark_warehouse').as_uri())
             .getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    startup = time.perf_counter() - start
    # Using a JVM SQL VALUES relation avoids Windows Hadoop native file helpers.
    # This is explicitly a small-data ingress; production ingestion should use
    # spark.read.parquet on the data lake, without collecting rows in the driver.
    def read_feature_table(name):
        path = OUT / 'spark_features' / f'{name}.csv'
        with path.open(encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            cols = list(reader.fieldnames)
            entries = list(reader)
        features = [c for c in cols if c.startswith('x')]
        rows = []
        for row in entries:
            rid = row['record_id'].replace("'", "''")
            values = [f"'{rid}'"] + [f'CAST({float(row[c]):.17g} AS DOUBLE)' for c in cols[1:]]
            rows.append('(' + ','.join(values) + ')')
        query = 'SELECT * FROM VALUES ' + ','.join(rows) + ' AS input(' + ','.join(cols) + ')'
        return spark.sql(query), features

    def train_predict(name):
        start = time.perf_counter()
        train, features = read_feature_table(name + '_train')
        test, test_features = read_feature_table(name + '_eval')
        assert features == test_features
        train.cache(); test.cache()
        ntr, nte = train.count(), test.count()
        ingress_seconds = time.perf_counter() - start
        pipe = Pipeline(stages=[
            VectorAssembler(inputCols=features, outputCol='features'),
            RandomForestRegressor(featuresCol='features', labelCol='label', numTrees=200,
                maxDepth=4, minInstancesPerNode=3, maxBins=32,
                featureSubsetStrategy='all', subsamplingRate=1.0, bootstrap=True, seed=SEED),
        ])
        start = time.perf_counter()
        fitted = pipe.fit(train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        rows = fitted.transform(test).select('record_id', 'label', 'prediction').collect()
        predict_seconds = time.perf_counter() - start
        train.unpersist(); test.unpersist()
        result = pd.DataFrame([r.asDict() for r in rows]).sort_values('record_id')
        print(f'Spark {name}: train={ntr}, eval={nte}, fit={fit_seconds:.2f}s', flush=True)
        return fitted, result, ingress_seconds, fit_seconds, predict_seconds

    try:
        cv_rows, cv_predictions = [], []
        for fold in range(4):
            fitted, data, ingress, fit, predict = train_predict(f'fold{fold}')
            cv_rows.append({'model': 'Random Forest Spark', 'fold': fold,
                **metrics(data.label, data.prediction), 'ingress_seconds': ingress,
                'fit_seconds': fit, 'predict_seconds': predict})
            cv_predictions.extend({'model': 'Random Forest Spark', 'fold': fold,
                'record_id': r.record_id, 'actual_cm': r.label, 'prediction_cm': r.prediction}
                for r in data.itertuples())
        fitted, predictions, ingress, fit, predict = train_predict('final')
        master = pd.read_csv(OUT / 'ml_dataset.csv', dtype={'kelurahan_id': str})
        predictions = predictions.merge(master[['record_id', 'kelurahan_id', 'split']], on='record_id', validate='one_to_one')
        assert predictions.split.eq('test').all()
        predictions = predictions.rename(columns={'label': 'actual_cm', 'prediction': 'prediction_cm'})
        predictions['residual_cm'] = predictions.actual_cm - predictions.prediction_cm
        predictions['model'] = 'Random Forest Spark'
        interval = cluster_bootstrap_mae(predictions.actual_cm.to_numpy(), predictions.prediction_cm.to_numpy(), predictions.kelurahan_id.to_numpy())
        cv = pd.DataFrame(cv_rows)
        row = {'model': 'Random Forest Spark', 'library': 'Spark MLlib',
               **metrics(predictions.actual_cm, predictions.prediction_cm),
               'cv_mae_mean_cm': float(cv.mae_cm.mean()), 'cv_mae_sd_cm': float(cv.mae_cm.std(ddof=1)),
               'fit_seconds': fit, 'predict_seconds': predict,
               'mae_ci_low_cm': interval[0], 'mae_ci_high_cm': interval[1], 'selected_by_local_cv': False}
        predictions.to_csv(OUT / 'test_predictions_spark.csv', index=False)
        cv.to_csv(OUT / 'cv_metrics_spark.csv', index=False)
        pd.DataFrame(cv_predictions).to_csv(OUT / 'cv_predictions_spark.csv', index=False)
        pd.DataFrame([row]).to_csv(OUT / 'test_metrics_spark.csv', index=False)
        rf = fitted.stages[-1]
        (OUT / 'models/spark_forest_trees.txt').write_text(rf.toDebugString, encoding='utf-8')
        save_json(OUT / 'spark_run.json', {'status': 'executed', 'spark_version': spark.version,
            'master': spark.sparkContext.master, 'startup_seconds': startup,
            'final_ingress_cache_seconds': ingress, 'feature_count': int(rf.numFeatures),
            'feature_importance': rf.featureImportances.toArray().tolist(),
            'parameters': {p.name: v for p, v in rf.extractParamMap().items()},
            'java_version': subprocess.run(['java', '-version'], capture_output=True, text=True).stderr.strip(),
            'input_method': 'small numeric feature tables -> JVM SQL VALUES; no Python workers or Hadoop native helpers',
            'artifact': 'tree text and parameters only; re-run script to rebuild Spark model',
            'timing_scope': 'local[2], one run; startup/ingress reported separately; not distributed speedup'})
        print(json.dumps(row, indent=2), flush=True)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
