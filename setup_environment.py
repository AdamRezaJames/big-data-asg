"""Prepare a local .venv with an available Python 3.12 and Java 17/21.

This bootstrap uses only the standard library and can be launched from Python 3.14.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def find_python312():
    candidates = [sys.executable,
        str(Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')]
    for command in ['python3.12', 'python3', 'python']:
        candidate = shutil.which(command)
        if candidate:
            candidates.append(candidate)
    launcher = shutil.which('py')
    if launcher:
        inventory = subprocess.run([launcher, '-0p'], capture_output=True, text=True)
        for line in inventory.stdout.splitlines():
            if '3.12' in line:
                match = re.search(r'([A-Za-z]:\\.*python(?:w)?\.exe)', line, re.I)
                if match:
                    candidates.append(match.group(1).strip())
    for path in dict.fromkeys(candidates):
        if not Path(path).is_file():
            continue
        result = subprocess.run([path, '-c', 'import sys,json; print(json.dumps(list(sys.version_info[:2])))'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            try:
                if json.loads(result.stdout) == [3, 12]:
                    return path
            except json.JSONDecodeError:
                pass
    raise SystemExit('Python 3.12 was not found. Install Python 3.12, reopen the terminal, then run this setup again.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check prerequisites without creating an environment or installing packages.')
    args = parser.parse_args()
    python = find_python312()
    java = shutil.which('java')
    if not java:
        raise SystemExit('Java was not found. Install JDK 17 and reopen the terminal.')
    version = subprocess.run([java, '-version'], capture_output=True, text=True)
    message = version.stderr + version.stdout
    match = re.search(r'version "(\d+)', message)
    if version.returncode or not match or int(match.group(1)) not in [17, 21]:
        raise SystemExit('This project uses Spark 4.0.1 with Java 17 or 21. Install JDK 17 to match the tested setup.')
    print('Python 3.12:', python, flush=True)
    print(message.splitlines()[0], flush=True)
    if args.check:
        print('Prerequisite check passed. No packages installed.')
        return
    venv = ROOT / '.venv'
    interpreter = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not interpreter.exists():
        subprocess.run([python, '-m', 'venv', str(venv)], cwd=ROOT, check=True)
    actual = subprocess.run([str(interpreter), '-c', 'import sys; print(".".join(map(str,sys.version_info[:2])))'],
                            capture_output=True, text=True, check=True)
    if actual.stdout.strip() != '3.12':
        raise SystemExit('The existing .venv uses another Python version. Rename that environment, then run setup again.')
    subprocess.run([str(interpreter), '-m', 'pip', 'install', '-r', str(ROOT / 'requirements.txt')], cwd=ROOT, check=True)
    print('\nEnvironment ready. Run:')
    print(r'.\.venv\Scripts\python.exe scripts\run.py' if os.name == 'nt' else '.venv/bin/python scripts/run.py')


if __name__ == '__main__':
    main()
