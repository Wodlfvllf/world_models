import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
import subprocess
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def source(cell):
    lines = cell["source"]
    if isinstance(lines, str):
        return lines
    return "".join(lines) if any(line.endswith("\n") for line in lines) else "\n".join(lines)


def notebook():
    return json.loads((ROOT / "notebooks/run_prerequisites.ipynb").read_text())


def native_function(name):
    tree = ast.parse((ROOT / "notebooks/run_prerequisites_marimo.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "marimo-cell", "exec"), namespace)
    return namespace[name]


def bundle(tmp_path, members=None):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for name, text in (members or {"scripts/molab_setup.py": "# stub", "pyproject.toml": "# test"}).items():
            output.writestr(name, text)
    archive.with_suffix(".zip.sha256").write_text(hashlib.sha256(archive.read_bytes()).hexdigest())
    return archive


def deploy(format, archive, project, execute=True):
    if format == "json":
        cell = next(c for c in notebook()["cells"] if "# Deploy only" in source(c))
        exec(compile(source(cell), "deploy-cell", "exec"), {
            "SOURCE_ZIP": archive, "PROJECT": project, "EXECUTE": execute})
    else:
        class Mo:
            @staticmethod
            def md(text):
                return text
            @staticmethod
            def stop(condition, message):
                if condition:
                    raise RuntimeError(message)
        native_function("deploy")(execute, project, archive, Mo)


@pytest.mark.parametrize("format", ["json", "marimo"])
def test_deployment_and_safe_reexecution(tmp_path, format):
    archive = bundle(tmp_path)
    project = tmp_path / "remote"
    deploy(format, archive, project)
    deploy(format, archive, project)
    assert (project / "scripts/molab_setup.py").is_file()
    (project / "scripts/molab_setup.py").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        deploy(format, archive, project)


@pytest.mark.parametrize("format", ["json", "marimo"])
def test_missing_receipt_has_actionable_error(tmp_path, format):
    archive = bundle(tmp_path)
    archive.with_suffix(".zip.sha256").unlink()
    with pytest.raises(FileNotFoundError, match="SOURCE_ZIP=None"):
        deploy(format, archive, tmp_path / "remote")


@pytest.mark.parametrize("format", ["json", "marimo"])
def test_git_deployment_pinned_and_non_destructive(tmp_path, monkeypatch, format):
    project = tmp_path / "remote"
    pinned = "864974efee26a5d09edf24d67f9e8aa3ed4f30b3"
    commands = []
    def fake_run(argv, **kwargs):
        commands.append(argv)
        assert kwargs["check"] and kwargs["timeout"] > 0
        if argv[1] == "init":
            (project / ".git").mkdir()
        if "checkout" in argv:
            (project / "scripts").mkdir()
            (project / "scripts/molab_setup.py").write_text("# pinned script")
    dirty = False
    def fake_output(argv, **kwargs):
        return (" M tracked.py" if dirty else "") if "status" in argv else pinned + "\n"
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "check_output", fake_output)
    deploy(format, None, project)
    assert len(commands) == 4
    assert commands[2][-1] == pinned and "--depth" in commands[2]
    deploy(format, None, project)
    assert len(commands) == 4
    dirty = True
    with pytest.raises(ValueError, match="changed"):
        deploy(format, None, project)
    assert (project / "scripts/molab_setup.py").read_text() == "# pinned script"


@pytest.mark.parametrize("format", ["json", "marimo"])
def test_git_refuses_existing_non_checkout(tmp_path, format):
    project = tmp_path / "existing"
    project.mkdir()
    with pytest.raises(FileExistsError, match="NEW PROJECT"):
        deploy(format, None, project)


@pytest.mark.parametrize("format", ["json", "marimo"])
@pytest.mark.parametrize("case", ["disabled", "tampered", "existing", "traversal", "protected"])
def test_deployment_safety(tmp_path, format, case):
    members = {"scripts/molab_setup.py": "stub"}
    if case == "traversal":
        members["../escape"] = "bad"
    if case == "protected":
        members["runs/old/data.json"] = "bad"
    archive = bundle(tmp_path, members)
    project = tmp_path / "remote"
    if case == "tampered":
        archive.with_suffix(".zip.sha256").write_text("0" * 64)
    if case == "existing":
        project.mkdir()
    with pytest.raises((RuntimeError, ValueError, FileExistsError)):
        deploy(format, archive, project, execute=case != "disabled")
    assert not (tmp_path / "escape").exists()
    assert not (project / "scripts/molab_setup.py").exists()


def test_notebook_syntax_and_sequential_dependencies():
    cells = notebook()["cells"]
    for cell in cells:
        if cell["cell_type"] == "code":
            compile(source(cell), "notebook-cell", "exec")
    tree = ast.parse((ROOT / "notebooks/run_prerequisites_marimo.py").read_text())
    previous = "preflight_done"
    for name, token in [("lewm_readout", "lewm_readout_done"), ("lewm_prediction", "lewm_prediction_done"),
                        ("dino_readout", "dino_readout_done"), ("dino_prediction", "dino_prediction_done"),
                        ("recorded_data", "recorded_data_done"), ("lewm_recorded", "lewm_recorded_done"),
                        ("dino_recorded", "dino_recorded_done"), ("verification", "verified")]:
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert previous in {a.arg for a in function.args.args}
        assert any(isinstance(n, ast.keyword) and n.arg == "after" and n.value.id == previous
                   for n in ast.walk(function))
        assert token in ast.unparse(function)
        previous = token


def test_hosted_environment_and_manifest(tmp_path, monkeypatch):
    module = runpy.run_path(str(ROOT / "scripts/molab_setup.py"))
    project = tmp_path / "cloud"
    (project / "configs").mkdir(parents=True)
    (project / "configs/pilot.json").write_text('{"seed": 3}')
    data = tmp_path / "previous-data"
    data.mkdir()
    (data / "manifest-development.json").write_text('{"config": {"seed": 3}}')
    commands = []
    def fake_stream(argv, cwd):
        commands.append(argv)
        python = project / ".venv/bin/python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.symlink_to(sys.executable)
    monkeypatch.setitem(module["prepare"].__globals__, "stream", fake_stream)
    python, actual_data = module["prepare"](project, data)
    assert commands == [["bash", "scripts/bootstrap.sh"]]
    assert python == str(project / ".venv/bin/python")
    assert actual_data == str(data)
    module["prepare"](project, data, existing_python=python)
    assert len(commands) == 1  # Reuses without rebuilding or resolving away the venv.
    (data / "manifest-development.json").write_text('{"config": {"seed": 4}}')
    with pytest.raises(ValueError, match="configuration"):
        module["prepare"](project, data, existing_python=python)


def test_workflow_retains_virtualenv_and_stops_out_of_order(tmp_path):
    module = runpy.run_path(str(ROOT / "scripts/notebook_workflow.py"))
    for name in ("scripts/run_diagnostics.py", "src/wm_audit/diagnostics.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    config = tmp_path / "config.json"
    config.write_text('{}')
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest-development.json").write_text('{"config": {}}')
    workflow = module["Workflow"](tmp_path, data, config, "new", execute=True, trust_checkpoint=True)
    assert workflow.python == python
    with pytest.raises(RuntimeError, match="sequence"):
        workflow.diagnostic("lewm", "readout", after="preflight")
    commands = []
    workflow._run = commands.append
    assert workflow.setup() == "setup"
    assert all(argv[0] == str(python) for argv in commands)
    assert "ensurepip" in commands[0][-1]
    with pytest.raises(RuntimeError, match="sequence"):
        workflow.setup()


def test_setup_stream_reports_failure(tmp_path, capsys):
    stream = runpy.run_path(str(ROOT / "scripts/molab_setup.py"))["stream"]
    stream([sys.executable, "-c", "print('visible progress')"], tmp_path)
    assert "visible progress" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="NOT started"):
        stream([sys.executable, "-c", "raise SystemExit(7)"], tmp_path)


def test_source_bundle_excludes_private_and_generated_files(tmp_path):
    package = runpy.run_path(str(ROOT / "scripts/package_molab.py"))["package"]
    archive = package(ROOT, tmp_path / "source.zip")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert "scripts/notebook_workflow.py" in names
        assert "scripts/molab_setup.py" in names
        assert "notebooks/run_prerequisites_marimo.py" in names
        exported = json.loads(z.read("notebooks/molab_cells.json"))
        for cell in exported["cells"]:
            assert cell["metadata"]["language"] in ("python", "markdown")
            assert cell["metadata"]["id"]
        assert "running.logs" not in names
        assert not any(n.startswith(("cache/", "runs/", ".venv/", ".git/")) for n in names)
    assert archive.with_suffix(".zip.sha256").read_text().split()[0] == hashlib.sha256(archive.read_bytes()).hexdigest()