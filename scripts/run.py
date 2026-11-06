"""Run preprocessing, local/Spark models, graph analysis, figures and evaluation."""
import os
import subprocess
import sys
from common import ROOT, setup


def main():
    setup()
    required = ['banjir_jakarta_timur_2020.csv', 'batas_kelurahan_dki.geojson',
                'road_nodes.csv', 'road_segments.csv', 'road_edges.csv']
    for name in required:
        if not (ROOT / 'data/silver' / name).is_file():
            raise SystemExit(f'Missing dataset: data/silver/{name}')
    stages = ['prepare_data.py', 'train_local.py', 'train_spark.py',
              'graph_analysis.py', 'make_figures.py', 'evaluate.py', 'show_results.py']
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    for name in stages:
        print(f'\nRunning {name}', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts' / name)], cwd=ROOT, env=env, check=True)
    print('\nFinished. Tables and models: results/\nFigures: results/figures/', flush=True)


if __name__ == '__main__':
    main()
