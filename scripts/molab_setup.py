"""Hosted-only environment/data preparation; no dependency on the Molab kernel packages."""
from pathlib import Path
import json
import os
import queue
import signal
import subprocess
import threading
import time


def stream(argv, project):
    """Stream setup output, retain a log, and stop the child on cell cancellation."""
    project = Path(project)
    messages = queue.Queue()
    started = time.monotonic()
    with (project / "molab-setup.log").open("a") as log:
        def emit(text):
            print(text, end="", flush=True)
            log.write(text)
            log.flush()
        emit(f"\nSETUP: {argv}\n")
        proc = subprocess.Popen(argv, cwd=project, env=dict(os.environ, PYTHONUNBUFFERED="1"),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", start_new_session=True)
        def read():
            try:
                for line in proc.stdout:
                    messages.put(line)
            finally:
                messages.put(None)
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        try:
            while True:
                try:
                    text = messages.get(timeout=30)
                except queue.Empty:
                    emit(f"[setup] RUNNING {time.monotonic()-started:.0f}s; waiting for subprocess output\n")
                    continue
                if text is None:
                    break
                emit(text)
            if proc.wait():
                raise RuntimeError("Hosted setup failed; inspect molab-setup.log. Experiments have NOT started.")
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            reader.join(timeout=2)
            proc.stdout.close()


def prepare(project, data, *, existing_python=None, backup=None, device="cuda"):
    """Use the previous project's interpreter or bootstrap an isolated CUDA env.

    Never deletes old data or environments. An optional receipt-verified backup
    is restored to a fresh nested directory, without replacing the updated source.
    """
    project = Path(project).resolve()
    python = Path(existing_python) if existing_python else project / ".venv/bin/python"
    if existing_python and not python.is_file():
        raise FileNotFoundError(f"Selected interpreter is missing: {python}; use None for fresh bootstrap")
    if not python.is_file():
        stream(["bash", "scripts/bootstrap.sh"] + (["--cpu"] if device == "cpu" else []), project)
    if backup:
        destination = project / "runs/restored-pilot"
        stream([str(python), "-u", "scripts/inspect_world_backup.py", "restore",
                str(Path(backup).resolve()), str(destination)], project)
        data = destination / "runs/pilot-gpu-01/data"
    else:
        data = Path(data)
        if not data.is_absolute():
            data = project / data
    if not (data / "manifest-development.json").is_file():
        raise FileNotFoundError(f"Pilot data missing at {data}. Set DATA to the existing/restored data directory, "
                                "or supply BACKUP_ZIP and its .zip.sha256 receipt. No new data was generated.")
    config = json.loads((project / "configs/pilot.json").read_text())
    if json.loads((data / "manifest-development.json").read_text()).get("config") != config:
        raise ValueError("Pilot data configuration mismatch; refusing to silently use different experiments")
    print(f"Hosted environment ready: {python}\nExisting pilot data: {data}", flush=True)
    return str(python), str(data.resolve())