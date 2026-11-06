"""Publication-style static PNG/SVG figures made only from saved experiment outputs."""
import os
from pathlib import Path
from common import ROOT, OUT, FIG, setup
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import json

BLUE, TEAL, ORANGE, PURPLE, GRAY = '#22628e', '#18867b', '#d28435', '#7859a6', '#79818b'


def save(fig, name):
    fig.savefig(FIG / (name + '.png'), dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(FIG / (name + '.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    setup()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
        'figure.constrained_layout.use': True})
    data = pd.read_csv(OUT / 'ml_dataset.csv')
    summary = json.loads((OUT / 'dataset_summary.json').read_text())
    local = json.loads((OUT / 'local_run.json').read_text())
    metrics = pd.concat([pd.read_csv(OUT / 'test_metrics_local.csv'), pd.read_csv(OUT / 'test_metrics_spark.csv')], ignore_index=True)
    preds = pd.concat([pd.read_csv(OUT / 'test_predictions_local.csv'), pd.read_csv(OUT / 'test_predictions_spark.csv')], ignore_index=True)
    graph_summary = json.loads((OUT / 'graph_summary.json').read_text())
    labels = {'Baseline median': 'Median', 'Baseline mean': 'Mean', 'Ridge': 'Ridge',
              'Random Forest sklearn': 'RF sklearn', 'Random Forest Spark': 'RF Spark'}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    for s, color in [('train', BLUE), ('test', ORANGE)]:
        values = data[data.split == s].target_cm
        ax[0].hist(values, bins=np.arange(0, 401, 25), color=color, alpha=.65, label=f'{s} (n={len(values)})')
    ax[0].set(xlabel='Batas atas ketinggian air laporan (cm)', ylabel='Jumlah baris laporan', title='Distribusi target menurut split')
    ax[0].legend(frameon=False)
    counts = data.groupby(['month', 'split']).size().unstack(fill_value=0).reindex(range(1, 13), fill_value=0)
    x = np.arange(1, 13)
    ax[1].bar(x-.18, counts.train, .36, color=BLUE, label='Latih')
    ax[1].bar(x+.18, counts.test, .36, color=ORANGE, label='Uji')
    ax[1].set(xticks=x, xlabel='Bulan pada periode laporan 2020', ylabel='Jumlah baris laporan', title='Cakupan bulan tidak merata')
    ax[1].legend(frameon=False)
    fig.suptitle('Dataset ML: 153 laporan, 61 kelurahan; bukan sampel hari banjir/tidak banjir', fontsize=11)
    save(fig, 'gambar-4-3-dataset-ml')

    fig, ax = plt.subplots(figsize=(10, 6.3))
    ax.set(xlim=(0, 10), ylim=(0, 7)); ax.axis('off')
    def box(x, y, w, h, content, color=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.07', facecolor='#f1f5f8', edgecolor=color, linewidth=1.4))
        ax.text(x+w/2, y+h/2, content, ha='center', va='center', fontsize=10, linespacing=1.5)
    def arrow(a, b):
        ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '-|>', 'color': '#526271', 'lw': 1.5})
    box(2.2, 5.9, 5.6, .8, '153 laporan historis + lokasi referensi kelurahan\nTarget: batas atas ketinggian air (cm)')
    box(2.2, 4.55, 5.6, .8, 'GroupShuffleSplit, seed 42\n113 latih / 40 uji; kelurahan terpisah')
    arrow((5, 5.9), (5, 5.4))
    box(.35, 2.85, 4.1, 1.1, 'Latih: GroupKFold 4 fold\nFit imputer dan one-hot hanya pada fold latih\nBaseline, Ridge, RF sklearn, RF Spark')
    box(5.55, 2.85, 4.1, 1.1, 'Uji ditahan selama pemilihan model\nTransformer final di-fit pada 113 baris\nTidak memasukkan ketinggian ke fitur', TEAL)
    arrow((3.5, 4.55), (2.4, 4)); arrow((6.5, 4.55), (7.6, 4))
    box(.35, 1.2, 4.1, 1, 'Pemilihan model lokal: MAE CV terendah\nRF Spark sebagai pembanding terpisah\nParameter tetap; tidak menala pada data uji')
    box(5.55, 1.2, 4.1, 1, 'Evaluasi pada 40 laporan uji\nMAE, RMSE, R²; bootstrap kelurahan\nSimpan prediksi, model lokal, dan metadata', TEAL)
    arrow((2.4, 2.85), (2.4, 2.25)); arrow((7.6, 2.85), (7.6, 2.25)); arrow((4.5, 1.7), (5.5, 1.7))
    ax.text(5, .3, 'Eksperimen retrospektif pada kelurahan yang tidak masuk data latih; bukan prakiraan banjir.', ha='center', fontsize=9, color='#56616c')
    save(fig, 'gambar-5-1-pipeline-ml')

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(metrics))
    for a, col, title in [(ax[0], 'mae_cm', 'MAE (cm): lebih kecil lebih baik'), (ax[1], 'rmse_cm', 'RMSE (cm): sensitif pada galat besar')]:
        colors = [GRAY, GRAY, PURPLE, TEAL, BLUE]
        a.bar(x, metrics[col], color=colors)
        a.set(xticks=x, xticklabels=[labels[m] for m in metrics.model], ylabel='cm', title=title, ylim=(0, metrics[col].max()*1.2))
        a.tick_params(axis='x', rotation=20)
        for i, v in enumerate(metrics[col]): a.text(i, v+1, f'{v:.2f}', ha='center', fontsize=9)
    fig.suptitle('Evaluasi pada 40 laporan dari 13 kelurahan uji yang sama', fontsize=11)
    save(fig, 'gambar-5-2-evaluasi-model')

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.4))
    for ax, name in zip(axes.flat, ['Baseline median', 'Ridge', 'Random Forest sklearn', 'Random Forest Spark']):
        subset = preds[preds.model == name]
        ax.scatter(subset.actual_cm, subset.prediction_cm, alpha=.72, s=30, color=TEAL if 'sklearn' in name else BLUE)
        ax.plot([0, 370], [0, 370], '--', color=GRAY, lw=1)
        ax.set(xlim=(0, 370), ylim=(0, 370), xlabel='Ketinggian tercatat (cm)', ylabel='Prediksi (cm)', title=name)
    fig.suptitle('Prediksi vs nilai tercatat. Garis putus-putus menunjukkan prediksi tepat', fontsize=11)
    save(fig, 'gambar-5-3-prediksi-aktual')

    importance = pd.read_csv(OUT / 'permutation_importance.csv')
    chosen = preds[preds.model == local['selected_by_cv']]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].barh(importance.feature_group, importance.mae_increase_cm, xerr=importance.permutation_sd_cm, color=TEAL, capsize=3)
    ax[0].axvline(0, color=GRAY, lw=.8)
    ax[0].set(xlabel='Kenaikan MAE setelah pengacakan (cm)', title='Diagnostik kelompok fitur RF sklearn')
    ax[1].scatter(chosen.prediction_cm, chosen.residual_cm, color=BLUE, alpha=.7)
    ax[1].axhline(0, color=GRAY, ls='--')
    ax[1].set(xlabel='Prediksi (cm)', ylabel='Tercatat − prediksi (cm)', title='Residual model pilihan CV')
    fig.suptitle('Diagnostik data uji. Tidak digunakan untuk menala model atau mengklaim sebab-akibat', fontsize=10)
    save(fig, 'gambar-5-4-diagnostik-model')

    nodes = pd.read_csv(ROOT / 'data/silver/road_nodes.csv').set_index('osm_node_id')
    segments = pd.read_csv(ROOT / 'data/silver/road_segments.csv')
    coordinates = nodes[['longitude', 'latitude']]
    lines = np.stack([coordinates.loc[segments.from_node].to_numpy(), coordinates.loc[segments.to_node].to_numpy()], axis=1)
    known = segments.exposure_index.notna().to_numpy()
    top = pd.read_csv(OUT / 'graph_top_nodes.csv')
    fig, ax = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={'width_ratios': [1.15, 1]})
    ax[0].add_collection(LineCollection(lines[~known], colors='#bbc0c5', linewidths=.32, rasterized=True))
    lc = LineCollection(lines[known], cmap='YlOrRd', linewidths=.48, rasterized=True)
    lc.set_array(segments.loc[known, 'exposure_index'].to_numpy()); lc.set_clim(0, 1)
    ax[0].add_collection(lc); ax[0].autoscale()
    ax[0].scatter(top.longitude.head(5), top.latitude.head(5), s=30, color=BLUE, zorder=4)
    for i, r in enumerate(top.head(5).itertuples(), 1):
        ax[0].annotate(str(i), (r.longitude, r.latitude), xytext=(4, 4), textcoords='offset points', fontsize=9, color=BLUE)
    ax[0].set(xlabel='Longitude WGS84', ylabel='Latitude WGS84', title='Graf jalan dan paparan historis kelurahan')
    ax[0].set_aspect(1 / np.cos(np.deg2rad(-6.25)))
    fig.colorbar(lc, ax=ax[0], label='Indeks paparan historis (bukan kedalaman ruas)', shrink=.75)
    values = top.head(8).iloc[::-1]
    ax[1].barh([str(n) for n in values.osm_node_id], values.sampled_betweenness, color=BLUE)
    ax[1].set(xlabel='Betweenness ternormalisasi, aproksimasi', ylabel='ID node OSM', title='Simpul dengan nilai tertinggi (k=32)')
    fig.suptitle('Graf kotak studi. Abu-abu = tidak memiliki profil yang cocok. Titik biru = 5 simpul teratas', fontsize=10)
    fig.supxlabel('© OpenStreetMap contributors. Laporan banjir Satu Data Jakarta 2020. Bukan peta kondisi akses aktual.', fontsize=9)
    save(fig, 'gambar-6-1-graf-jalan')

    routes = pd.read_csv(OUT / 'graph_routes.csv')
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for col, destination in enumerate(['BIDARA CINA', 'KAMPUNG MELAYU']):
        subset = routes[routes.destination == destination]
        baseline = subset[subset['lambda'] == 0].iloc[0]
        for u, color, label in [(0, ORANGE, 'Tidak diketahui diberi 0'), (1, BLUE, 'Tidak diketahui diberi 1')]:
            line = pd.concat([subset[(subset['lambda'] > 0) & (subset.unknown_penalty == u)], pd.DataFrame([baseline])]).sort_values('lambda')
            axes[0, col].plot(line['lambda'], line.distance_m, 'o-', color=color, label=label)
            axes[1, col].plot(line['lambda'], line.unknown_length_m, 'o-', color=color, label=label)
        axes[0, col].set(title='Cawang → ' + destination.title(), ylabel='Jarak rute (m)')
        axes[1, col].set(xlabel='Parameter penalti λ', ylabel='Panjang berindikator null (m)', xticks=[0, 1, 2, 4])
        axes[1, col].set_ylim(-15, 700)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle('Sensitivitas rute terhadap λ dan perlakuan data yang belum diketahui', fontsize=11)
    save(fig, 'gambar-6-2-sensitivitas-rute')
    print(f'Saved 7 figures in PNG and SVG: {FIG}')


if __name__ == '__main__':
    main()
