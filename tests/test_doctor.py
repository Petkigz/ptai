"""
The pre-flight check: one screen, the real reason, a non-zero exit.

Borrowed from the sibling project's `npm run doctor`. The point is not the list
of checks - it is that a launcher can ask "will this actually run?" and get an
answer in two seconds, instead of opening windows that die and leaving the
operator to guess.

What has to hold here:

  * a FAIL is a FAIL (exit 1) and a WARN never blocks (exit 0) - a warning is
    something to know, not something to stop for;
  * the checks READ. Nothing trades, writes a trade, or calls a venue: a health
    check with side effects is a second way for the system to do something the
    operator did not ask for;
  * the runner check keeps the LF trap closed, because an LF-only .bat is the
    failure that has cost this project two rounds on the operator's machine.
"""

import os
import socket

import pytest

from src.ptai import doctor


def test_it_reports_every_check_it_claims_to():
    checks = doctor.run_checks()
    names = [c.name for c in checks]
    for expected in ("Python version", "Required packages", "data/ writable",
                     "Database", "Console port", "Runner", "Venues"):
        assert expected in names, f"{expected} is not checked at all: {names}"


def test_every_check_carries_a_reason_and_a_next_step():
    """A FAIL with no remedy is a wall; the remedy is the point of the tool."""
    for check in doctor.run_checks():
        assert check.detail, f"{check.name} says nothing"
        if check.status == doctor.FAIL:
            assert check.remedy, (
                f"{check.name} failed and told the operator nothing to do")


def test_a_clean_environment_exits_zero(tmp_path, monkeypatch):
    """
    A healthy install must not fail on anything.

    `--network` is off and the model probe is skipped, so this is a genuinely
    local question: can this machine start PTAI?
    """
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "ptai.db"))
    monkeypatch.setenv("PTAI_DATA_DIR", str(tmp_path / "data"))
    checks = doctor.run_checks(root=tmp_path, port=_free_port(),
                               db_path=tmp_path / "ptai.db")
    # The runner is the one thing a temp root cannot have; everything else is
    # about this machine and must pass.
    failures = [c for c in checks
                if c.status == doctor.FAIL and c.name != "Runner"]
    assert not failures, [c.as_dict() for c in failures]


def test_a_missing_dependency_is_a_failure_with_the_pip_command(monkeypatch):
    """The most common install failure, and the fix is one command."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
        else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("No module named 'pandas'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    checks = doctor.run_checks()
    failing = [c for c in checks if c.name == "Required packages"]
    assert failing and failing[0].status == doctor.FAIL
    assert "pandas" in failing[0].detail
    assert "pip install" in failing[0].remedy


def test_a_taken_port_is_a_failure_that_names_the_port():
    """
    "Port already in use" is the most common silent launcher failure on a
    machine that runs more than one project, and the fix is one line in one
    file - so the check says which port.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        check = doctor._port_check(port)
    assert check.status == doctor.FAIL
    assert str(port) in check.detail
    assert "PTAI_DASHBOARD_PORT" in check.remedy


def test_an_lf_only_runner_is_a_failure(tmp_path):
    """
    cmd.exe breaks on an LF-only .bat (the window pops and closes), which is
    exactly what happened on the operator's PC. Checking costs microseconds.
    """
    (tmp_path / "run_ptai.bat").write_bytes(b"@echo off\necho hi\npause\n")
    check = doctor._runner_check(tmp_path)
    assert check.status == doctor.FAIL
    assert "LF" in check.detail

    (tmp_path / "run_ptai.bat").write_bytes(b"@echo off\r\necho hi\r\npause\r\n")
    assert doctor._runner_check(tmp_path).status == doctor.PASS


def test_more_than_one_runner_is_a_warning_not_a_failure(tmp_path):
    """
    One entry point is the design, and extra launchers are how the pieces drift
    apart - but a spare script is not a reason to refuse to start.
    """
    for name in ("run_ptai.bat", "start_old.bat"):
        (tmp_path / name).write_bytes(b"@echo off\r\n")
    check = doctor._runner_check(tmp_path)
    assert check.status == doctor.WARN
    assert "2 .bat" in check.detail


def test_the_report_says_ready_or_says_what_to_fix():
    ok = doctor.format_report([doctor.Check("A", doctor.PASS, "fine")])
    assert "Ready to run" in ok and "0 fail" in ok

    bad = doctor.format_report([
        doctor.Check("A", doctor.PASS, "fine"),
        doctor.Check("B", doctor.FAIL, "broken", "do the thing")])
    assert "Fix the FAIL item(s)" in bad
    assert "do the thing" in bad, "the remedy was not printed"


def test_main_exits_nonzero_only_on_a_failure(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "run_checks",
                        lambda **kw: [doctor.Check("A", doctor.PASS, "fine")])
    assert doctor.main([]) == 0

    monkeypatch.setattr(doctor, "run_checks",
                        lambda **kw: [doctor.Check("A", doctor.WARN, "hmm")])
    assert doctor.main(["--quiet"]) == 0, "a warning must not stop the launcher"

    monkeypatch.setattr(doctor, "run_checks",
                        lambda **kw: [doctor.Check("A", doctor.FAIL, "broken")])
    assert doctor.main([]) == 1


def test_it_reads_and_does_not_trade(tmp_path, monkeypatch):
    """
    The check must not be a second way for the system to act.

    A doctor that placed an order, or wrote to the outcome log, would be worse
    than no doctor at all - and it would be invisible, because nobody reads the
    health check as a risk.
    """
    monkeypatch.setenv("PTAI_DB", str(tmp_path / "ptai.db"))
    db = tmp_path / "ptai.db"
    doctor.run_checks(root=tmp_path, db_path=db, port=_free_port())
    import sqlite3
    with sqlite3.connect(str(db)) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert tables, "the doctor did not even create the schema"
        for table in ("trades", "trade_outcomes", "orders"):
            if table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                assert count == 0, (
                    f"the doctor wrote {count} row(s) to {table} - it is "
                    f"supposed to read")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]
