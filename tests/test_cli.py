"""CLI integration contracts; artifact-dependent execution is tested separately."""
from argparse import Namespace

from wm_audit import cli, simulator


def test_simulator_uses_downloaded_vendored_package(tmp_path, monkeypatch):
    monkeypatch.setattr(simulator, "PushTSimulator", lambda source: source)
    assert cli._sim(Namespace(cache=tmp_path)) == tmp_path / "sources/point/stable-worldmodel"