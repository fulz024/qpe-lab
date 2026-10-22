#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
if [[ "${1:-}" == --proof-tools ]]; then
    cd .tools/bubblewrap-0.11.0
    apt-get download libcap-dev
    for package in libcap-dev_*.deb; do dpkg-deb -x "$package" cap-dev; done
    printf '%s\n' '#define PACKAGE_STRING "bubblewrap 0.11.0"' '#define ENABLE_REQUIRE_USERNS 1' > config.h
    gcc -O2 -D_GNU_SOURCE -I. -Icap-dev/usr/include bubblewrap.c bind-mount.c network.c utils.c -Wl,-l:libcap.so.2 -o bwrap
    ./bwrap --unshare-all --ro-bind / / --size 1048576 --tmpfs /tmp --die-with-parent /bin/true
    exit 0
fi
bash scripts/setup_wsl.sh
if [[ ! -x .tools/bubblewrap/usr/bin/bwrap ]]; then
    mkdir -p .cache/bubblewrap-package .tools/bubblewrap
    (
        cd .cache/bubblewrap-package
        apt-get download bubblewrap
        for package in bubblewrap_*.deb; do
            dpkg-deb -x "$package" "$project_root/.tools/bubblewrap"
        done
    )
fi
.tools/bubblewrap/usr/bin/bwrap --unshare-all --ro-bind / / --proc /proc --dev /dev --die-with-parent /bin/true
export UV_PROJECT_ENVIRONMENT="$project_root/.venv-lab"
export UV_LINK_MODE=copy
.tools/uv-linux/bin/uv sync --frozen --python 3.12.15
.venv-lab/bin/python - <<'PY'
import tempfile
from qpe_lab.evaluation.sandbox import _training_filter
with tempfile.TemporaryFile() as policy:
    _training_filter(policy)
print("Training seccomp policy available")
PY
.venv-lab/bin/python -m qpe_lab inspect benchmarks/dfthc/fe2s2/lookup.task.json
