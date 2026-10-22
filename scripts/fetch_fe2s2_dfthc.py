"""Fetch just one small ZIP member, pin it, and prepare the angle lookup task.

Uses HTTP range requests; refuses servers that try to return the whole 14.8 GB ZIP.
Only this pinned official numerical pickle is converted. Candidates use JSON only.
"""

import argparse
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from qpe_lab.contracts import digest
from qpe_lab.tasks.data.angles import write_task
from qpe_lab.tasks.data.dfthc import SOURCE_MEMBER, SOURCE_URL, load_official_source

ROOT = Path(__file__).resolve().parents[1]


class RangeZip(io.RawIOBase):
    def __init__(self):
        req = urllib.request.Request(SOURCE_URL, headers={"Range": "bytes=-65557"})
        with urllib.request.urlopen(req, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError("Server ignored Range; refusing full ZIP download")
            self.size = int(response.headers["Content-Range"].split("/")[-1])
            self.tail = response.read(65558)
        self.pos = 0

    def seek(self, offset, whence=0):
        self.pos = offset + (self.size if whence == 2 else self.pos if whence == 1 else 0)
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size - self.pos)
        if n <= 0:
            return b""
        if n > 32 * 1024**2:
            raise RuntimeError("Refusing oversized ZIP range")
        start = self.pos
        self.pos += n
        if start >= self.size - len(self.tail):
            offset = start - self.size + len(self.tail)
            return self.tail[offset : offset + n]
        request = urllib.request.Request(
            SOURCE_URL + f"&start={start}", headers={"Range": f"bytes={start}-{start + n - 1}"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206 or not response.headers["Content-Range"].startswith(
                f"bytes {start}-"
            ):
                raise RuntimeError("Server did not honor exact byte range")
            data = response.read(n + 1)
        if len(data) != n:
            raise RuntimeError("ZIP range size mismatch")
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="Use a cached copy of the pinned source member")
    args = parser.parse_args()
    if args.source:
        raw = args.source.read_bytes()
    else:
        with zipfile.ZipFile(RangeZip()) as archive:
            info = archive.getinfo(SOURCE_MEMBER)
            if info.file_size > 1_000_000:
                raise ValueError("Source member unexpectedly large")
            raw = archive.read(SOURCE_MEMBER)
    factors = load_official_source(raw)
    directory = ROOT / "benchmarks/dfthc/fe2s2"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "factors.json"
    path.write_text(json.dumps(factors, indent=2) + "\n")
    task, table = write_task(path, directory)
    manifest = {
        key: factors[key] for key in ("source_url", "source_member", "source_sha256", "system")
    }
    manifest.update(
        factors_sha256=digest(path),
        angle_table_sha256=task["table_sha256"],
        task_sha256=digest(directory / "lookup.task.json"),
        scope="two-body shared-vector angle lookup; one-body/SOS/walk integration pending",
    )
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "task": str(directory / "lookup.task.json"),
                "vectors": table["valid_vectors"],
                "word_bits": task["word_bits"],
                "max_quantized_vector_l2_residual": table["max_quantized_vector_l2_residual"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
