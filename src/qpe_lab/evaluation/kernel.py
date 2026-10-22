"""Independent lookup cross-check through the original C execution backend."""

from uncompute.evalcore import RegSpec, Task, TestCase, ops_hash, run_task


def sample_lookup(task, stream_path, output, jobs=1):
    """Independent C-kernel cross-check; exact acceptance comes from our checker."""

    class Lookup(Task):
        name = task.name
        seed_domain = b"qpe-lab/lookup/v1"
        num_tests = len(task.words) * 8

        def schema(self):
            return [RegSpec(task.address_bits), RegSpec(task.word_bits)]

        def gen_test_indexed(self, shot, xof):
            address = shot // 8
            # Deterministic z patterns include zero, ones, and pseudo-random targets.
            import hashlib

            z = int.from_bytes(
                hashlib.shake_256(str(shot).encode()).digest((task.word_bits + 7) // 8), "little"
            )
            z &= (1 << task.word_bits) - 1
            if shot % 8 == 0:
                z = 0
            elif shot % 8 == 1:
                z = (1 << task.word_bits) - 1
            return TestCase(inputs=[address, z], expected=[address, z ^ task.words[address]])

    with open(stream_path, "rb") as f:
        return run_task(
            Lookup(),
            str(task.path.parent),
            f,
            ops_hash(stream_path),
            out_dir=str(output),
            stream_path=stream_path,
            jobs=jobs,
        )
