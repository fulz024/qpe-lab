#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
mkdir -p runs

for executable in python3 git gcc make; do
    command -v "$executable" >/dev/null || { echo "Missing prerequisite: $executable" >&2; exit 1; }
done
git -c core.autocrlf=false submodule update --init --recursive
python3 - <<'PY'
import hashlib
import json
import pathlib
import subprocess
root = pathlib.Path.cwd()
pin = json.loads((root / 'config/upstream.json').read_text())
upstream = root / 'third_party/uncompute'
head = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
if head != pin['commit']:
    raise SystemExit(f'Upstream commit mismatch: {head}')
if hashlib.sha256((upstream / 'uv.lock').read_bytes()).hexdigest() != pin['uv_lock_sha256']:
    raise SystemExit('Upstream dependency lock mismatch.')
if subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
    raise SystemExit('Upstream has local changes; preserve them and resolve before baseline reproduction.')
PY

uv_bin="$project_root/.tools/uv-linux/bin/uv"
if [[ ! -x "$uv_bin" ]]; then
    # Bootstrap outside the system Python environment (also works with PEP 668).
    python3 -m venv "$project_root/.bootstrap/uv"
    "$project_root/.bootstrap/uv/bin/python" -m pip install \
        --target "$project_root/.tools/uv-linux" 'uv==0.13.1'
fi
[[ "$("$uv_bin" --version)" == 'uv 0.13.1'* ]] || { echo 'Expected uv 0.13.1' >&2; exit 1; }
export UV_PROJECT_ENVIRONMENT="$project_root/.venv-wsl"
export UV_LINK_MODE=copy
python_version="$(python3 -c "import json; print(json.load(open('config/upstream.json'))['python_version'])")"
"$uv_bin" sync --project "$project_root/third_party/uncompute" --python "$python_version" --frozen
make -C "$project_root/third_party/uncompute" PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python" kernel
"$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import sys
from uncompute.evalcore.kernel import KERNEL_ABI, kernel_available
if not kernel_available():
    raise SystemExit('C evaluator kernel is unavailable.')
print(f'Ready: Python {sys.version.split()[0]}, C kernel ABI {KERNEL_ABI}')
PY
