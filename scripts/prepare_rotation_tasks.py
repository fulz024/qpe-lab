"""Derive rotation contracts from existing angle tables, keeping lookup inputs fixed."""

import json
from pathlib import Path

from qpe_lab.contracts import digest
from qpe_lab.tasks.data.angles import write_rotation_task


def main():
    root = Path(__file__).resolve().parents[1]
    toy = root / "benchmarks/dfthc/toy/rotation"
    toy.mkdir(parents=True, exist_ok=True)
    # Three spatial modes, two 3-bit angles; the last address is padded identity.
    table = {
        "schema": "qpe.angle-table.v1",
        "words_hex": ["0x11", "0x2b", "0x3e", "0x0"],
        "angle_bits": 3,
        "angles_per_word": 2,
        "valid_vectors": 3,
        "packing": "little-endian angle words in descending elimination-pair order",
    }
    table_path = toy / "angle-table.json"
    table_path.write_text(json.dumps(table, indent=2) + "\n")
    task = {
        "schema": "qpe.lookup.v1",
        "name": "toy-givens-3m-b3",
        "table": "angle-table.json",
        "table_sha256": digest(table_path),
        "address_bits": 2,
        "word_bits": 6,
        "max_qubits": 64,
    }
    (toy / "lookup.task.json").write_text(json.dumps(task, indent=2) + "\n")
    paths = [
        write_rotation_task(toy / "lookup.task.json", spin_copies=1),
        write_rotation_task(root / "benchmarks/dfthc/fe2s2/lookup.task.json"),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
