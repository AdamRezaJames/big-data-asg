"""Inspect previously executed results using only the Python standard library."""
import argparse
import csv
import json
import os
from pathlib import Path
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open-figure', action='store_true', help='Open the saved model comparison PNG.')
    args = parser.parse_args()
    path = ROOT / 'results/all_test_metrics.csv'
    if not path.exists():
        raise SystemExit('Saved results are missing. Run scripts/run.py first.')
    with path.open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    print('MODEL RESULTS: 40 held-out reports from 13 kelurahan')
    print(f'{"Model":26} {"MAE (cm)":>10} {"RMSE (cm)":>11} {"R2":>9}')
    for row in rows:
        print(f'{row["model"]:26} {float(row["mae_cm"]):10.2f} {float(row["rmse_cm"]):11.2f} {float(row["r2"]):9.4f}')
    graph = json.loads((ROOT / 'results/graph_summary.json').read_text(encoding='utf-8'))
    print(f'\nGRAPH: {graph["nodes"]:,} nodes, {graph["directed_edges"]:,} directed edges')
    print(f'Weak components: {graph["weak_components"]}, strong components: {graph["strong_components"]}')
    print('\nFigures:', ROOT / 'results/figures')
    print('These are saved experiment results. This command does not retrain models.')
    if args.open_figure:
        image = ROOT / 'results/figures/gambar-5-2-evaluasi-model.png'
        if os.name == 'nt':
            os.startfile(str(image))
        else:
            webbrowser.open(image.as_uri())


if __name__ == '__main__':
    main()
