"""Standard-library execution bridge for the Marimo/JSON diagnostic notebooks.

Always uses the project's interpreter, not the hosted notebook kernel. A workflow
owns a fresh output directory; there is deliberately no implicit resume.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import shutil
import signal
import subprocess
import threading
import time
import zipfile


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_report(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    if report.get("schema") != "wm_audit.diagnostic.v1" or report.get("confirmatory") is not False:
        raise ValueError("Not an exploratory diagnostic report")
    if not report.get("artifacts") or not report.get("records"):
        raise ValueError("Incomplete diagnostic report")
    for name, expected in report["artifacts"].items():
        path = directory / name
        if Path(name).name != name or name.startswith(".") or path.is_symlink():
            raise ValueError("Unsafe report artifact path")
        if sha256(path) != expected:
            raise ValueError(f"Artifact checksum mismatch: {path}")
    return report


class Workflow:
    def __init__(self, project, data, config, run_id, *, python=None, device="cuda",
                 trust_checkpoint=False, recorded=False, download_recorded=False,
                 h5=None, episodes=128, install=True, execute=False):
        if not execute:
            raise RuntimeError("Review settings, then set EXECUTE = True to authorize this run")
        if not trust_checkpoint:
            raise ValueError("Both-model notebook requires explicit TRUST_PINNED_DINO = True")
        if device not in ("cpu", "cuda") or episodes < 1:
            raise ValueError("Invalid device or episode count")
        if download_recorded and not recorded:
            raise ValueError("DOWNLOAD_RECORDED requires RUN_RECORDED")
        if not run_id or Path(run_id).name != run_id or run_id.startswith("."):
            raise ValueError("RUN_ID must be a simple, non-hidden directory name")
        self.project = Path(project).expanduser().resolve()
        self.data = self._path(data)
        self.config = self._path(config)
        # Resolving a venv's python symlink selects the base interpreter and loses
        # its site-packages. Keep the executable path inside the chosen venv.
        executable = Path(python or ".venv/bin/python").expanduser()
        self.python = Path(os.path.abspath(executable if executable.is_absolute() else self.project / executable))
        self.cache = self.project / "cache"
        self.h5 = self._path(h5 or "cache/datasets/lewm-pusht/pusht_expert_train.h5")
        self.recorded, self.download_recorded = recorded, download_recorded
        self.episodes, self.install = episodes, install
        for path in (self.project / "scripts/run_diagnostics.py", self.project / "src/wm_audit/diagnostics.py",
                     self.python, self.config, self.data / "manifest-development.json"):
            if not path.is_file():
                raise FileNotFoundError(f"Required input missing: {path}; upload updated source / restore data first")
        config_value = json.loads(self.config.read_text())
        if json.loads((self.data / "manifest-development.json").read_text()).get("config") != config_value:
            raise ValueError("Configuration differs from original data generation manifest")
        if recorded and not self.h5.is_file() and not download_recorded:
            raise FileNotFoundError("Recorded H5 missing; set DOWNLOAD_RECORDED=True explicitly or supply H5")
        default_h5 = self.cache / "datasets/lewm-pusht/pusht_expert_train.h5"
        if download_recorded and self.h5 != default_h5:
            raise ValueError("Downloader writes the default cache H5; use that path or disable download")
        self.output = self.project / "runs" / run_id
        self.output.mkdir(parents=True, exist_ok=False)
        self.prefix = [str(self.python), "-u", "-m", "wm_audit.cli", "--cache", str(self.cache),
                       "--device", device, "--trust-checkpoint"]
        self.last = "configured"
        self.completed = []
        self.failed = False
        self.reports = []
        self.commands = []
        self._write_plan()
        print(f"Configured fresh run: {self.output}\nDevice: {device}\nExisting data: {self.data}", flush=True)

    def _path(self, path):
        path = Path(path).expanduser()
        return (path if path.is_absolute() else self.project / path).resolve()

    def _write_plan(self):
        (self.output / "diagnostics-plan.json").write_text(json.dumps({
            "note": "Exploratory prerequisites only; no editor refit or confirmation",
            "data": str(self.data), "config": str(self.config),
            "config_sha256": sha256(self.config),
            "generation_manifest_sha256": sha256(self.data / "manifest-development.json"),
            "recorded_enabled": self.recorded, "commands": self.commands,
            "completed": self.completed, "failed": self.failed,
        }, indent=2))

    def _run(self, argv):
        self.commands.append(argv)
        self._write_plan()
        env = dict(os.environ, PYTHONUNBUFFERED="1", WM_AUDIT_PROGRESS="1",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        lines = queue.Queue()
        begin = time.monotonic()
        with (self.output / "console.log").open("a") as log:
            def emit(text):
                print(text, end="", flush=True)
                log.write(text)
                log.flush()

            emit(f"\n>>> {shlex.join(argv)}\n")
            process = subprocess.Popen(argv, cwd=self.project, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors="replace",
                                       bufsize=1, start_new_session=True)
            def read_lines():
                try:
                    for line in process.stdout:
                        lines.put(line)
                finally:
                    lines.put(None)

            reader = threading.Thread(target=read_lines, daemon=True)
            reader.start()
            try:
                while True:
                    try:
                        line = lines.get(timeout=30)
                    except queue.Empty:
                        emit(f"[notebook] RUNNING | elapsed {time.monotonic() - begin:.0f}s | waiting for child output\n")
                        continue
                    if line is None:
                        break
                    emit(line)
                code = process.wait()
                if code:
                    raise RuntimeError(f"Command exited {code}; preserve {self.output}; remaining cells must not run")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                reader.join(timeout=2)
                process.stdout.close()

    def _step(self, name, after, action):
        if self.failed or after != self.last or name in self.completed:
            raise RuntimeError("Cells must run once in sequence; use a NEW RUN_ID after an interrupted/repeated run")
        print(f"\n=== START {name} ===", flush=True)
        try:
            action()
        except BaseException:
            self.failed = True
            self._write_plan()
            raise
        self.completed.append(name)
        self.last = name
        self._write_plan()
        print(f"=== DONE {name} ===", flush=True)
        return name

    def setup(self, after="configured"):
        def action():
            if self.install:
                self._run([str(self.python), "-u", "-c",
                           "import importlib.util\nif importlib.util.find_spec('pip') is None:\n"
                           "    import ensurepip\n    ensurepip.bootstrap()"])
                self._run([str(self.python), "-u", "-m", "pip", "install", "-e", str(self.project)])
            # Runs using the project interpreter; validates all selected roots before GPU work.
            check = ("from pathlib import Path; from wm_audit.provenance import read_config; "
                     "from wm_audit.diagnostics import _roster; "
                     f"_roster(read_config({str(self.config)!r}), Path({str(self.data)!r}), "
                     "('scorer','validation','pilot')); print('Input roster verified')")
            self._run([str(self.python), "-u", "-c", check])
        return self._step("setup", after, action)

    def preflight(self, after):
        def action():
            self._run(self.prefix + ["download"])
            self._run(self.prefix + ["preflight", "--output", str(self.output / "preflight.json")])
        return self._step("preflight", after, action)

    def diagnostic(self, model, kind, after):
        if model not in ("lewm", "dinowm") or kind not in ("readout", "prediction-paired", "prediction-recorded"):
            raise ValueError("Unknown diagnostic")
        def action():
            if kind == "prediction-recorded" and not self.recorded:
                print("SKIPPED recorded prediction: RUN_RECORDED=False (not a successful experiment)", flush=True)
                return
            directory = self.output / model / kind
            command = "diagnose-readout" if kind == "readout" else "diagnose-prediction"
            argv = self.prefix + [command, "--config", str(self.config), "--model", model,
                                  "--output", str(directory)]
            argv += (["--h5", str(self.h5), "--episodes", str(self.episodes)] if kind == "prediction-recorded"
                     else ["--data", str(self.data)])
            self._run(argv)
            verify_report(directory)
            self.reports.append(directory)
            print(f"Verified report and plots: {directory / 'index.html'}", flush=True)
        return self._step(f"{model}/{kind}", after, action)

    def prepare_recorded(self, after):
        def action():
            if not self.recorded:
                print("SKIPPED recorded data (explicit opt-in required)", flush=True)
                return
            if self.download_recorded:
                self.cache.mkdir(parents=True, exist_ok=True)
                if not self.h5.is_file() and shutil.disk_usage(self.cache).free < 65_000_000_000:
                    raise RuntimeError("At least 65 GB free disk required before recorded download/extraction")
                self._run(self.prefix + ["download", "--dataset"])
            if not self.h5.is_file():
                raise FileNotFoundError(self.h5)
        return self._step("recorded-data", after, action)

    def finish(self, after):
        def action():
            expected = 6 if self.recorded else 4
            if len(self.reports) != expected:
                raise RuntimeError("Missing diagnostic stages")
            for directory in self.reports:
                verify_report(directory)
            print(f"Verified {expected} diagnostic reports. No scientific pass/fail inferred.", flush=True)
        return self._step("verified", after, action)

    def archive(self, after):
        if after != "verified" or self.last != after:
            raise RuntimeError("Verify results before exporting")
        archive = self.output.with_name(self.output.name + "-results.zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(self.output.rglob("*")):
                if path.is_symlink():
                    raise ValueError("Refusing to export symlinks")
                if path.is_file():
                    bundle.write(path, path.relative_to(self.output.parent))
        checksum = sha256(archive)
        with archive.with_suffix(".zip.sha256").open("x") as receipt:
            receipt.write(f"{checksum}  {archive.name}\n")
        print(f"Results-only archive: {archive}\nSHA256: {checksum}\nKeep the original data/source backup separately.", flush=True)
        return archive