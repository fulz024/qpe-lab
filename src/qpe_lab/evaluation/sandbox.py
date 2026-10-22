"""Bounded jobs. Bubblewrap isolates generator files/processes/network on Linux."""

import ctypes
import ctypes.util
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_job(command, cwd, log_path, timeout, env, *, pass_fds=(), heartbeat=None):
    started = time.monotonic()
    command = [
        "/usr/bin/prlimit",
        f"--as={4 * 1024**3}",
        f"--fsize={64 * 1024**2}",
        "--nofile=256",
        "--",
        *command,
    ]
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=pass_fds,
        )
        timed_out = False
        try:
            while True:
                if heartbeat is not None:
                    heartbeat()
                remaining = max(0.01, timeout) - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    proc.wait(timeout=min(2, remaining) if heartbeat is not None else remaining)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - started >= timeout:
                        timed_out = True
                        break
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Kill surviving children even when the main generator exited normally.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    return {
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "seconds": time.monotonic() - started,
    }


def minimal_env():
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "HOME": "/tmp",
    }


def _training_filter(output):
    """Export a fail-closed native seccomp filter, installed by bubblewrap.

    Training generators have one process, one writable output file and a log.
    Denying process creation makes the address-space cap job-wide. Anonymous
    file/IPC storage and namespace manipulation are also unavailable.
    """
    library = ctypes.util.find_library("seccomp")
    if not library:
        raise RuntimeError("Training isolation requires libseccomp; no fallback")
    lib = ctypes.CDLL(library, use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.seccomp_export_bpf.restype = ctypes.c_int
    context = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("Cannot allocate seccomp filter")
    try:
        for name in (
            "fork",
            "vfork",
            "clone",
            "clone3",
            "memfd_create",
            "shmget",
            "shmat",
            "msgget",
            "semget",
            "io_uring_setup",
            "mount",
            "umount2",
            "unshare",
            "setns",
            "ptrace",
            "process_vm_writev",
            "bpf",
            "userfaultfd",
            "socket",
            "socketpair",
            "open_by_handle_at",
            "fsopen",
            "fsmount",
            "move_mount",
            "open_tree",
        ):
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number < 0 or lib.seccomp_rule_add(context, 0x00050001, number, 0) != 0:
                raise RuntimeError(f"Cannot restrict syscall {name}")
        if lib.seccomp_export_bpf(context, output.fileno()) != 0:
            raise RuntimeError("Cannot export seccomp filter")
        output.seek(0)
    finally:
        lib.seccomp_release(context)


def generate(
    root, candidate, task, out, timeout, *, reference=None, workspace=None, heartbeat=None
):
    bwrap = root / ".tools/bubblewrap/usr/bin/bwrap"
    if not bwrap.is_file():
        raise RuntimeError(
            "Bubblewrap unavailable; run scripts/setup-lab.ps1. No silent unsandboxed fallback."
        )
    command = [
        str(bwrap),
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for path in (
        Path("/usr"),
        Path("/lib"),
        Path("/lib64"),
        Path(sys.base_prefix),
        Path(sys.prefix),
    ):
        if path.exists():
            command += ["--ro-bind", str(path), str(path)]
    for source, dest in (
        (root / "src", "/app/src"),
        (root / "third_party/uncompute", "/opt/uncompute"),
        (candidate.parent, "/candidate"),
        (task.path.parent, "/task"),
    ):
        command += ["--ro-bind", str(source), dest]
    if workspace is not None:
        command += ["--ro-bind", str(workspace), "/workspace"]
        for name in ("docs", "candidates", "benchmarks", "AGENTS.md", "README.md"):
            command += ["--ro-bind", str(root / name), "/app/" + name]
    if reference is None:
        command += ["--bind", str(out), "/out"]
    else:
        # A single file bind is the only writable filesystem surface. Neither
        # sidecar files nor unbounded tmpfs can bypass the per-file limit.
        (out / "circuit.ops").touch(exist_ok=False)
        command += [
            "--ro-bind",
            str(out),
            "/out",
            "--bind",
            str(out / "circuit.ops"),
            "/out/circuit.ops",
            "--ro-bind",
            str(reference),
            "/reference.json",
            "--remount-ro",
            "/tmp",
            "--remount-ro",
            "/dev",
            "--remount-ro",
            "/",
            "--cap-drop",
            "ALL",
        ]
    command += ["--chdir", "/out"]
    env = minimal_env() | {
        "PYTHONPATH": "/app/src:/opt/uncompute",
        "QPE_TASK_PATH": "/task/" + task.path.name,
        "UNCOMPUTE_OPS_PATH": "/out/circuit.ops",
    }
    if reference is not None:
        env["QPE_REFERENCE_PATH"] = "/reference.json"
    if workspace is not None:
        env["PYTHONPATH"] += ":/workspace"
    for key, value in env.items():
        command += ["--setenv", key, value]
    executable = [sys.executable, "-s", "/candidate/" + candidate.name]
    if reference is None:
        return run_job(
            command + executable,
            root,
            out.parent / "generator.log",
            timeout,
            minimal_env(),
            heartbeat=heartbeat,
        )
    with tempfile.TemporaryFile() as policy:
        _training_filter(policy)
        command += ["--seccomp", str(policy.fileno()), *executable]
        return run_job(
            command,
            root,
            out.parent / "generator.log",
            timeout,
            minimal_env(),
            pass_fds=(policy.fileno(),),
            heartbeat=heartbeat,
        )
