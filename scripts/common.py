"""Shared paths, snapshot extraction and experiment utilities.

Road data: Copyright OpenStreetMap contributors, ODbL 1.0.
https://www.openstreetmap.org/copyright
"""
import hashlib
import gzip
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
FIG = OUT / 'figures'
SEED = 42
NUMERIC = ['month_sin', 'month_cos', 'longitude', 'latitude']
CATEGORICAL = ['kecamatan']


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    ensure_datasets()


def ensure_datasets():
    """Restore the exact compressed input snapshots in a portable repository copy."""
    manifest = ROOT / 'data/dataset_manifest.json'
    if not manifest.exists():
        return
    for item in json.loads(manifest.read_text(encoding='utf-8')):
        if item.get('compression') != 'gzip':
            continue
        target = (ROOT / item['file']).resolve()
        archive = (ROOT / item['distribution_file']).resolve()
        if ROOT not in target.parents or ROOT not in archive.parents:
            raise ValueError('Dataset paths must remain within the project folder.')
        if target.exists():
            if target.stat().st_size != item['bytes'] or sha256(target) != item['sha256']:
                raise ValueError(f'Dataset differs from the recorded snapshot: {target.name}')
            continue
        if sha256(archive) != item['distribution_sha256']:
            raise ValueError(f'Compressed dataset checksum mismatch: {archive.name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.part')
        print(f'Extracting {archive.name} ...', flush=True)
        with gzip.open(archive, 'rb') as source, temporary.open('wb') as output:
            shutil.copyfileobj(source, output)
        if temporary.stat().st_size != item['bytes'] or sha256(temporary) != item['sha256']:
            temporary.unlink()
            raise ValueError(f'Extracted dataset checksum mismatch: {target.name}')
        temporary.replace(target)
