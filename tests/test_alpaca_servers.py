"""Unit tests for the local ASCOM Remote + QHY Alpaca launcher."""

from pathlib import Path

import pytest

from obs_utils import alpaca_servers as servers


class _Process:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


@pytest.fixture(autouse=True)
def _clear_server_processes():
    servers._server_processes.clear()
    yield
    servers._server_processes.clear()


def test_uses_local_observatory_servers_only_for_windows_loopback(monkeypatch):
    monkeypatch.setattr(servers, "_is_windows", lambda: True)
    assert servers.uses_local_observatory_alpaca_servers(
        "localhost:11111", "127.0.0.1:11112"
    )
    assert not servers.uses_local_observatory_alpaca_servers(
        "192.168.1.4:11111", "localhost:11112"
    )
    assert not servers.uses_local_observatory_alpaca_servers(
        "localhost:11111", None
    )


def test_launcher_reuses_two_ready_servers_without_spawning(monkeypatch):
    monkeypatch.setattr(servers, "_is_windows", lambda: True)
    monkeypatch.setattr(servers, "_server_is_ready", lambda endpoint: True)
    spawned = []
    monkeypatch.setattr(servers.subprocess, "Popen", lambda *args, **kwargs: spawned.append(args))

    status = servers.start_observatory_alpaca_servers()

    assert not status.ascom_started
    assert not status.qhy_started
    assert spawned == []


def test_launcher_starts_both_unavailable_servers_then_confirms_readiness(monkeypatch, tmp_path):
    monkeypatch.setattr(servers, "_is_windows", lambda: True)
    ready = {"localhost:11111": False, "localhost:11112": False}
    monkeypatch.setattr(servers, "_server_is_ready", lambda endpoint: ready[endpoint])
    starts = []

    def launch_ascom(executable):
        starts.append(("ascom", executable))
        ready["localhost:11111"] = True
        return _Process()

    def launch_qhy(root, python):
        starts.append(("qhy", root, python))
        ready["localhost:11112"] = True
        return _Process()

    monkeypatch.setattr(servers, "_launch_ascom_remote", launch_ascom)
    monkeypatch.setattr(servers, "_launch_qhy_server", launch_qhy)

    status = servers.start_observatory_alpaca_servers(
        qhy_root=tmp_path,
        python_executable=Path("python.exe"),
    )

    assert status.ascom_started and status.qhy_started
    assert [item[0] for item in starts] == ["ascom", "qhy"]
    assert set(servers._server_processes) == {"ASCOM Remote", "QHY Alpaca"}


def test_launcher_does_not_duplicate_a_server_it_already_started(monkeypatch, tmp_path):
    monkeypatch.setattr(servers, "_is_windows", lambda: True)
    ready = {"localhost:11111": False, "localhost:11112": True}
    monkeypatch.setattr(servers, "_server_is_ready", lambda endpoint: ready[endpoint])
    servers._server_processes["ASCOM Remote"] = _Process()

    def await_servers(*args, **kwargs):
        ready["localhost:11111"] = True

    monkeypatch.setattr(servers, "_await_servers", await_servers)
    monkeypatch.setattr(
        servers,
        "_launch_ascom_remote",
        lambda executable: pytest.fail("should not spawn a duplicate ASCOM server"),
    )

    status = servers.start_observatory_alpaca_servers(qhy_root=tmp_path)

    assert not status.ascom_started
    assert not status.qhy_started
