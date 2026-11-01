import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def introduction():
    import marimo as mo
    mo.md("""
    # Frozen world-model experiments — sequential GPU notebook

    **Run this notebook on Molab, with the GPU attached.** Upload the supplied
    `world-models-molab-source.zip` and its `.zip.sha256` receipt to Molab.
    The notebook installs that code in a NEW hosted project directory, prepares
    the remote Python environment, then calls the experiment scripts there.
    No local VS Code interpreter or local filesystem is used.

     1. Review Cell 2: point SOURCE_ZIP at your Molab upload, DATA at the previous
         pilot data, and choose a **new** run ID. All paths are on Molab.
         On a fresh Molab session, supply BACKUP_ZIP plus its checksum receipt to
         restore the saved pilot. No previous pilot needs to be rerun.
    2. Set `TRUST_PINNED_DINO=True` only if you consent to the pinned released
       DINO pickle. Set `EXECUTE=True` to authorize execution.
    3. Use **Run all** (or run cells top to bottom). Dependency tokens enforce
       sequential execution, even with Marimo's reactive scheduling.

    Default: **four** experiments — actual-image readout and paired prediction,
    for LeWM then DINO. Set `RUN_RECORDED=True` for **six** experiments. If its H5
    is absent, also opt into `DOWNLOAD_RECORDED=True` (~13 GB download, ~46 GB
    extracted; requires 65 GB free). Recorded checks use original appearance only.

    These are exploratory prerequisites, **not** a repeat pilot or confirmation.
    **Training cells fit physical readouts**; LeWM/DINO weights stay frozen.
    No editor refit or world-model training is performed. Model cache downloads
    are expected. Keep runtime auto-run disabled while
    editing settings. Interrupted/repeated runs require a **new RUN_ID**; there
    is no automatic resume and no overwriting existing results.
    """)
    return (mo,)


@app.cell
def settings():
    from pathlib import Path
    import importlib.util

    SOURCE_ZIP = Path("/marimo/world-models-molab-source.zip")  # Uploaded bundle.
    PROJECT = Path("/marimo/world_models_diagnostics")  # NEW directory, preserves the old project.
    DATA = "/marimo/world_models/runs/pilot-gpu-01/data"  # Existing remote pilot data.
    BACKUP_ZIP = None  # Fresh session: absolute path to uploaded backup ZIP + adjacent receipt.
    EXISTING_PYTHON = None  # None bootstraps remote CUDA env; or '/marimo/world_models/.venv/bin/python'.
    CONFIG = "configs/pilot.json"   # Must match the original data manifest.
    RUN_ID = "diagnostics-notebook-01"  # Change for every new attempt.
    DEVICE = "cuda"
    EXECUTE = False
    TRUST_PINNED_DINO = False
    INSTALL_UPDATED_PACKAGE = True
    RUN_RECORDED = False
    DOWNLOAD_RECORDED = False
    H5 = "cache/datasets/lewm-pusht/pusht_expert_train.h5"
    EPISODES = 128
    return (BACKUP_ZIP, CONFIG, DATA, DEVICE, DOWNLOAD_RECORDED, EPISODES, EXECUTE,
            EXISTING_PYTHON, H5, INSTALL_UPDATED_PACKAGE, PROJECT, SOURCE_ZIP, RUN_ID, RUN_RECORDED,
            TRUST_PINNED_DINO, importlib)


@app.cell
def deploy(EXECUTE, PROJECT, SOURCE_ZIP, mo):
    mo.stop(not EXECUTE, mo.md("**Not started.** Review Cell 2 and set EXECUTE=True."))
    import hashlib
    import stat
    import zipfile
    from pathlib import PurePosixPath

    # The source bundle contains the unpushed scripts; cloning old GitHub main is insufficient.
    _receipt = SOURCE_ZIP.with_suffix(".zip.sha256")
    _expected = _receipt.read_text().split()[0]
    with SOURCE_ZIP.open("rb") as _stream:
        _actual = hashlib.file_digest(_stream, "sha256").hexdigest()
    if _actual != _expected:
        raise ValueError("Source ZIP checksum mismatch; re-upload bundle and receipt")
    with zipfile.ZipFile(SOURCE_ZIP) as _archive:
        _names = _archive.namelist()
        if len(set(_names)) != len(_names) or "scripts/molab_setup.py" not in _names:
            raise ValueError("Invalid source bundle")
        if sum(_info.file_size for _info in _archive.infolist()) > 50_000_000:
            raise ValueError("Expected a small source-only bundle, not a data/model backup")
        for _info in _archive.infolist():
            _parts = PurePosixPath(_info.filename)
            if (_parts.is_absolute() or ".." in _parts.parts or "\\" in _info.filename
                    or ":" in _info.filename or _info.is_dir() or stat.S_ISLNK(_info.external_attr >> 16)
                    or _parts.parts[0] in {"runs", "cache", ".venv", ".git"}):
                raise ValueError("Unsafe source archive member")
        _marker = PROJECT / "MOLAB-SOURCE.sha256"
        if PROJECT.exists() or PROJECT.is_symlink():
            if PROJECT.is_symlink() or not _marker.is_file() or _marker.read_text().strip() != _actual:
                raise FileExistsError("PROJECT must be new, or a previous extraction of this exact bundle")
            for _name in _names:
                _target = PROJECT / _name
                if (_target.is_symlink() or not _target.resolve().is_relative_to(PROJECT.resolve())
                        or _target.read_bytes() != _archive.read(_name)):
                    raise ValueError("Hosted source changed; use a new PROJECT directory")
        else:
            PROJECT.mkdir(parents=True, exist_ok=False)
            for _name in _names:
                _target = PROJECT / _name
                _target.parent.mkdir(parents=True, exist_ok=True)
                _target.write_bytes(_archive.read(_name))
            _marker.write_text(_actual + "\n")
    source_ready = str(PROJECT)
    print(f"Verified source deployed on Molab: {source_ready}", flush=True)
    return (source_ready,)


@app.cell
def hosted_environment(BACKUP_ZIP, DATA, DEVICE, EXISTING_PYTHON, PROJECT, importlib, source_ready):
    _spec = importlib.util.spec_from_file_location("molab_setup", PROJECT / "scripts/molab_setup.py")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    hosted_python, hosted_data = _module.prepare(source_ready, DATA, existing_python=EXISTING_PYTHON,
                                                backup=BACKUP_ZIP, device=DEVICE)
    return (hosted_data, hosted_python)


@app.cell
def configure(CONFIG, DEVICE, DOWNLOAD_RECORDED, EPISODES, EXECUTE, H5,
              INSTALL_UPDATED_PACKAGE, PROJECT, hosted_data, hosted_python, RUN_ID, RUN_RECORDED,
              TRUST_PINNED_DINO, importlib, mo):
    mo.stop(not EXECUTE, mo.md("**Not started.** Review Cell 2 and set EXECUTE=True."))
    _spec = importlib.util.spec_from_file_location("notebook_workflow", PROJECT / "scripts/notebook_workflow.py")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    workflow = _module.Workflow(PROJECT, hosted_data, CONFIG, RUN_ID, python=hosted_python,
        device=DEVICE, trust_checkpoint=TRUST_PINNED_DINO, recorded=RUN_RECORDED,
        download_recorded=DOWNLOAD_RECORDED, h5=H5, episodes=EPISODES,
        install=INSTALL_UPDATED_PACKAGE, execute=EXECUTE)
    return (workflow,)


@app.cell
def install_environment(workflow):
    setup_done = workflow.setup()
    return (setup_done,)


@app.cell
def preflight(setup_done, workflow):
    preflight_done = workflow.preflight(after=setup_done)
    return (preflight_done,)


@app.cell
def lewm_readout(preflight_done, workflow):
    lewm_readout_done = workflow.diagnostic("lewm", "readout", after=preflight_done)
    return (lewm_readout_done,)


@app.cell
def lewm_prediction(lewm_readout_done, workflow):
    lewm_prediction_done = workflow.diagnostic("lewm", "prediction-paired", after=lewm_readout_done)
    return (lewm_prediction_done,)


@app.cell
def dino_readout(lewm_prediction_done, workflow):
    dino_readout_done = workflow.diagnostic("dinowm", "readout", after=lewm_prediction_done)
    return (dino_readout_done,)


@app.cell
def dino_prediction(dino_readout_done, workflow):
    dino_prediction_done = workflow.diagnostic("dinowm", "prediction-paired", after=dino_readout_done)
    return (dino_prediction_done,)


@app.cell
def recorded_data(dino_prediction_done, workflow):
    recorded_data_done = workflow.prepare_recorded(after=dino_prediction_done)
    return (recorded_data_done,)


@app.cell
def lewm_recorded(recorded_data_done, workflow):
    lewm_recorded_done = workflow.diagnostic("lewm", "prediction-recorded", after=recorded_data_done)
    return (lewm_recorded_done,)


@app.cell
def dino_recorded(lewm_recorded_done, workflow):
    dino_recorded_done = workflow.diagnostic("dinowm", "prediction-recorded", after=lewm_recorded_done)
    return (dino_recorded_done,)


@app.cell
def verification(dino_recorded_done, workflow):
    verified = workflow.finish(after=dino_recorded_done)
    return (verified,)


@app.cell
def results(mo, verified, workflow):
    _panels = [mo.md("## Verified diagnostic plots\nCompletion is not evidence that the hypothesis passed.")]
    if verified == "verified":
        for _directory in workflow.reports:
            _panels.append(mo.md(f"### {_directory.parent.name} / {_directory.name}"))
            for _image in sorted(_directory.glob("*.png")):
                _panels.append(mo.image(src=_image.read_bytes(), alt=_image.name))
    mo.vstack(_panels)
    return


@app.cell
def download(mo, verified, workflow):
    results_zip = workflow.archive(after=verified)
    mo.vstack([
        mo.md("## Download results\nIncludes reports, plots, readouts and logs. **Not** the input data, model cache or project source. Keep your original backup."),
        mo.download(data=results_zip.read_bytes(), filename=results_zip.name, label="Download diagnostic results ZIP"),
        mo.md("Next: inspect measurement accuracy and native prediction versus controls. Only then revisit clean editor qualification and a fresh, locked confirmation. Recorded H5 is pretraining data, not held-out generalization."),
    ])
    return (results_zip,)


if __name__ == "__main__":
    app.run()