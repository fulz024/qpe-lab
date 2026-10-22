"""One pinned official Fe2S2 source; no general-purpose pickle interface."""

import hashlib
import io
import pickle

import numpy as np

SOURCE_SHA256 = "781de90ecd9628b0aa5a09eb7935a312821c4d661b183d4fd61283a34ba0e0e4"
SOURCE_MEMBER = (
    "spinfree/best_solutions/Fe2S2-20o30e_14_15_5_ln10_ep1000_lrf10_tg977_te10_ndiff8"
    "_ln10_tg977_seed2000000.pickle"
)
SOURCE_URL = "https://zenodo.org/records/17066718/files/spinfree.zip?download=1"


def _jax_array(function, args, state, _aval):
    if function is not np._core.multiarray._reconstruct:
        raise ValueError("Unexpected pinned JAX array constructor")
    array = function(*args)
    array.__setstate__(state)
    return array


class _PinnedReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("jax._src.array", "_reconstruct_array"): _jax_array,
            ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy.core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        if (module, name) not in allowed:
            raise ValueError(f"Unapproved source constructor: {module}.{name}")
        return allowed[module, name]


def load_official_source(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("Official source bytes do not match the pinned specimen")
    record = _PinnedReader(io.BytesIO(raw)).load()
    params = record["params_best"]
    if params["U"].shape != (14, 15, 20) or params["W"].shape != (14, 15, 5):
        raise ValueError("Unexpected Fe2S2 factor dimensions")
    if any(not np.isfinite(value).all() for value in params.values()):
        raise ValueError("Nonfinite factor data")
    return {
        "schema": "qpe.dfthc.factors.v1",
        "source_url": SOURCE_URL,
        "source_member": SOURCE_MEMBER,
        "source_sha256": SOURCE_SHA256,
        "system": "Fe2S2-20o30e",
        "R": 14,
        "B": 15,
        "C": 5,
        "N": 20,
        "params": {key: value.tolist() for key, value in params.items()},
        "published_values": {key: value.tolist() for key, value in record["values"].items()},
        "scope": "official factors retained; current task uses two-body shared orbital vectors only",
    }
