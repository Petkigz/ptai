"""
PTAI doctor - the pre-flight check, before anything is started.

Borrowed from the sibling project (`avt-bot`), which has `npm run doctor`: one
command, one screen, PASS/WARN/FAIL for every reason the program could fail to
start, and a non-zero exit so a launcher can stop instead of pressing on.

Why it exists here: every hard failure this project has had on a user's machine
was a *pre*condition — Python not on PATH, dependencies missing or half
installed, a port already taken by something else, a database that cannot be
written, no model endpoint. Each of those produced a window that popped up and
closed, or a traceback thirty seconds into a cycle, and the operator was left
reading tea leaves. A check that runs in two seconds and prints the actual
reason is worth more than any amount of error handling sprinkled downstream.

The exit code is the contract: 0 means every check passed, 1 means at least one
FAILed. WARN never fails the run — a warning is something to know about, not
something that stops the agent working.

Nothing here trades, writes to the database, or talks to a venue. It reads.
"""

from __future__ import annotations

import os
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

# The third-party modules the running system actually imports. Kept as
# distribution names for the remedy line and import names for the check, because
# "pip install pydantic-settings" and "import pydantic_settings" are not the same
# string and a single list would print one of them wrong.
REQUIRED_IMPORTS = (
    ("loguru", "loguru"),
    ("pydantic", "pydantic"),
    ("pydantic_settings", "pydantic-settings"),
    ("httpx", "httpx"),
    ("rich", "rich"),
    ("typer", "typer"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
)
# Needed for live Polymarket orders and for the browser logins. Absent is a WARN,
# not a FAIL: paper mode needs neither, and that is the default.
OPTIONAL_IMPORTS = (
    ("py_clob_client_v2", "py-clob-client-v2"),
    ("playwright", "playwright"),
)


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    remedy: str = ""

    @property
    def ok(self) -> bool:
        return self.status != FAIL

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status,
                "detail": self.detail, "remedy": self.remedy}


def _python_check() -> Check:
    version = sys.version_info
    text = f"{version.major}.{version.minor}.{version.micro}"
    if version < (3, 10):
        return Check("Python version", FAIL, f"{text} - need 3.10 or newer",
                     "Install Python 3 from python.org and tick "
                     "'Add python.exe to PATH'.")
    return Check("Python version", PASS, f"{text} ({sys.executable})")


def _imports_check() -> List[Check]:
    checks: List[Check] = []
    missing: List[str] = []
    for module, distribution in REQUIRED_IMPORTS:
        try:
            __import__(module)
        except Exception as e:  # noqa: BLE001 - any import failure is a failure
            missing.append(f"{distribution} ({type(e).__name__})")
    if missing:
        checks.append(Check(
            "Required packages", FAIL, ", ".join(missing),
            "Run: python -m pip install --user -r requirements.txt"))
    else:
        checks.append(Check(
            "Required packages", PASS,
            f"all {len(REQUIRED_IMPORTS)} import"))
    optional_missing = []
    for module, distribution in OPTIONAL_IMPORTS:
        try:
            __import__(module)
        except Exception:  # noqa: BLE001
            optional_missing.append(distribution)
    if optional_missing:
        checks.append(Check(
            "Optional packages", WARN, ", ".join(optional_missing) + " not installed",
            "Only needed for live venue orders or browser logins; paper mode "
            "runs without them."))
    else:
        checks.append(Check("Optional packages", PASS, "all present"))
    return checks


def _env_check(root: Path) -> Check:
    env = root / ".env"
    example = root / ".env.example"
    if env.exists():
        return Check(".env file", PASS, str(env))
    if example.exists():
        return Check(".env file", WARN, "not created yet - built-in defaults are used",
                     f"Copy {example.name} to .env to change settings "
                     f"(the console can also pin the model without it).")
    return Check(".env file", WARN, "not present", "")


def _data_dir_check(data_dir: Path) -> Check:
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / f".doctor-{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as e:  # noqa: BLE001
        return Check("data/ writable", FAIL, f"{data_dir}: {type(e).__name__}: {e}",
                     "The agent's memory lives there. Check permissions, or a "
                     "cloud-sync folder holding the file open.")
    return Check("data/ writable", PASS, str(data_dir))


def _storage_check(db_path: Path) -> List[Check]:
    """
    The database, opened the way the console opens it.

    A failure here is the one that produces "every panel is empty" or a 500 with
    no explanation, so it is checked before anything else touches it.
    """
    checks: List[Check] = []
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from .storage.db import Storage  # noqa: WPS433 (deliberate late import)
    except Exception as e:  # noqa: BLE001
        return [Check("Database", FAIL, f"storage module did not import: "
                                        f"{type(e).__name__}: {e}",
                      "This is a broken installation or a half-copied folder.")]
    storage = None
    try:
        storage = Storage(db_path=str(db_path))
        row = storage.conn.execute("SELECT 1").fetchone()
        assert row is not None
        checks.append(Check("Database", PASS, f"{db_path} opens and answers"))
    except Exception as e:  # noqa: BLE001
        checks.append(Check(
            "Database", FAIL, f"{db_path}: {type(e).__name__}: {e}",
            "Delete nothing: rename the file and PTAI will build a fresh one. "
            "If it is locked, close the other PTAI window first."))
        return checks

    try:
        from .execution.capital import operator_mode
        mode = operator_mode(storage)
        equity = storage.get_bankroll()
        paper = storage.get_paper_bankroll()
        checks.append(Check(
            "Operator account", PASS,
            f"mode {mode}, real bankroll ${equity:.2f}, "
            f"paper bankroll ${paper:.2f}",
            "" if mode == "paper" else "Live mode: real capital can be deployed "
                                       "once a venue is funded and qualified."))
    except Exception as e:  # noqa: BLE001
        checks.append(Check("Operator account", WARN,
                            f"could not read: {type(e).__name__}: {e}"))

    # Has the agent ever run? An install that has never completed a cycle looks
    # identical to one whose runner is broken, and this is how you tell.
    try:
        last = storage.get_state("last_cycle_completed") or ""
        heartbeat = storage.get_state("last_heartbeat") or ""
        if last or heartbeat:
            checks.append(Check("Agent history", PASS,
                                f"last cycle {last or 'unknown'}, "
                                f"last heartbeat {heartbeat or 'unknown'}"))
        else:
            checks.append(Check(
                "Agent history", WARN, "no completed cycle has ever been recorded",
                "Expected on a fresh install. If you have been running PTAI, the "
                "runner is not reaching a cycle - run it once and re-check."))
    except Exception as e:  # noqa: BLE001
        checks.append(Check("Agent history", WARN,
                            f"could not read: {type(e).__name__}: {e}"))
    finally:
        try:
            if storage is not None:
                storage.close()
        except Exception:  # noqa: BLE001
            pass
    return checks


def _port_check(port: int, host: str = "127.0.0.1") -> Check:
    """
    Is the console's port free?

    "Port already in use" is the single most common reason a launcher fails
    silently on a machine that runs more than one project, and the fix is one
    line in one file - so the check says which port and how to change it.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.6)
    try:
        in_use = probe.connect_ex((host, port)) == 0
    except Exception as e:  # noqa: BLE001
        return Check("Console port", WARN, f"{port}: could not probe ({e})")
    finally:
        probe.close()
    if in_use:
        return Check("Console port", FAIL, f"{port} is already in use",
                     "Another program (or another PTAI) owns it. Set "
                     "PTAI_DASHBOARD_PORT to a free port at the top of "
                     "run_ptai.bat, then run it again.")
    return Check("Console port", PASS, f"{port} free")


def _runner_check(root: Path) -> Check:
    """
    One runner, and it still has Windows line endings.

    cmd.exe silently breaks on an LF-only .bat (the window pops and closes), and
    that exact failure has cost this project two rounds. Checking takes
    microseconds and the remedy is a save-as.
    """
    bats = sorted(p.name for p in root.glob("*.bat"))
    if not bats:
        return Check("Runner", FAIL, "no .bat file in the project root",
                     "run_ptai.bat is the one entry point and it is missing.")
    if len(bats) > 1:
        return Check("Runner", WARN, f"{len(bats)} .bat files: {', '.join(bats)}",
                     "One entry point is the design; extra launchers are how the "
                     "pieces drift apart.")
    bat = root / bats[0]
    try:
        raw = bat.read_bytes()
        crlf = raw.count(b"\r\n")
        bare_lf = raw.count(b"\n") - crlf
    except Exception as e:  # noqa: BLE001
        return Check("Runner", WARN, f"{bat.name}: {type(e).__name__}: {e}")
    if bare_lf:
        return Check("Runner", FAIL,
                     f"{bat.name} has {bare_lf} LF-only line(s)",
                     "Re-save it with Windows line endings, or run: "
                     "git rm --cached run_ptai.bat && git checkout run_ptai.bat")
    return Check("Runner", PASS, f"{bat.name} (CRLF)")


def _model_check(network: bool = False) -> Check:
    """
    Is a model endpoint configured - and, if asked, answering?

    No probe by default. `is_available()` is a blocking HTTP call, and a
    pre-flight check that hangs for the length of a network timeout is a worse
    failure than the one it is looking for. `--network` opts into the probe.

    WARN either way: PTAI forecasts without a model — it falls back to the
    market's own price and says so — but an operator who thinks they are running
    a model should know they are not.
    """
    try:
        from .config import get_settings  # noqa: WPS433
        settings = get_settings()
        host = (os.environ.get("LM_STUDIO_HOST") or settings.lm_studio_host)
        pinned = (os.environ.get("LM_STUDIO_MODEL")
                  or getattr(settings, "lm_studio_model", "") or "auto")
    except Exception as e:  # noqa: BLE001
        return Check("Model endpoint", WARN,
                     f"config did not load: {type(e).__name__}: {e}")
    if not network:
        return Check("Model endpoint", PASS,
                     f"configured: {host}, model {pinned} "
                     f"(not probed - use --network to test it)")
    try:
        from .llm.provider import LMStudioProvider  # noqa: WPS433
        provider = LMStudioProvider(host=host, model=pinned)
        if provider.is_available():
            return Check("Model endpoint", PASS, f"answering at {host}")
        return Check(
            "Model endpoint", WARN, f"nothing answering at {host}",
            "Optional. Start LM Studio and load a model, or leave it: PTAI "
            "forecasts from the market's own price and labels it as such.")
    except Exception as e:  # noqa: BLE001
        return Check("Model endpoint", WARN,
                     f"probe failed: {type(e).__name__}: {e}")


def _venues_check() -> Check:
    """
    What the venue layer can actually do, in one line.

    Reads the real registry the agent builds at startup, because "19 adapters"
    and "one venue that can place a real order" are different facts and only the
    second one is about money. Informational: the adapter count has never been a
    measure of anything.
    """
    try:
        from .agent.v3_loop import TradingAgentV3  # noqa: WPS433
        from .venues.inventory import build_inventory  # noqa: WPS433

        agent = TradingAgentV3(country_code="UG", dry_run=True)
        inventory = build_inventory(agent.venue_registry)
        counts = inventory["counts"]
        # Four numbers, because they answer four different questions and the
        # single "19 registered" answered none of them: what can be read now,
        # what is simulated, what could hold real money at all, and what has
        # never been built.
        detail = (
            f"{counts['registered']} registered: "
            f"{counts['readable_now']} readable now with no account, "
            f"{counts['paper_tradable']} paper-traded, "
            f"{counts['can_place_real_orders']} armed, "
            f"{counts['real_order_path']} with an order path at all, "
            f"{counts['no_client']} with no client written")
        armed = [row["label"] for row in inventory["venues"].values()
                 if row["can_place_real_orders"]]
        if armed:
            return Check("Venues", PASS, detail + f" (armed: {', '.join(armed)})")
        return Check(
            "Venues", PASS, detail,
            "Expected without venue credentials: everything runs in paper "
            "against the live markets, which is what earns the qualification "
            "that unlocks real capital.")
    except Exception as e:  # noqa: BLE001
        return Check("Venues", WARN,
                     f"registry did not build: {type(e).__name__}: {e}")


def run_checks(root: Optional[Path] = None, *, port: Optional[int] = None,
               db_path: Optional[Path] = None, network: bool = False,
               include_model: bool = True) -> List[Check]:
    """
    Every check, in the order a failure would bite.

    `root` is the project directory; everything else is derived from it unless
    given, so a caller (a test, the CLI, the console) can point it somewhere else
    without the checks inventing paths.
    """
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    data_dir = Path(os.environ.get("PTAI_DATA_DIR") or (root / "data"))
    if db_path is None:
        db_path = Path(os.environ.get("PTAI_DB") or (data_dir / "ptai.db"))
    if port is None:
        port = int(os.environ.get("PTAI_CONSOLE_PORT")
                   or os.environ.get("PTAI_DASHBOARD_PORT") or 8101)

    checks: List[Check] = [_python_check()]
    checks.extend(_imports_check())
    checks.append(_env_check(root))
    checks.append(_data_dir_check(data_dir))
    checks.extend(_storage_check(db_path))
    checks.append(_port_check(port))
    checks.append(_runner_check(root))
    if include_model:
        checks.append(_model_check(network=network))
    checks.append(_venues_check())
    return checks


def format_report(checks: List[Check]) -> str:
    """The screen: one line per check, then the verdict."""
    icons = {PASS: "[ok]", WARN: "[warn]", FAIL: "[FAIL]"}
    width = max((len(c.name) for c in checks), default=0)
    lines = ["", "PTAI doctor", "=" * (width + 70)]
    for check in checks:
        line = f"{icons.get(check.status, '?'):>7} {check.name:<{width}}  {check.detail}"
        lines.append(line.rstrip())
        if check.remedy and check.status in (WARN, FAIL):
            lines.append(f"{'':>9} -> {check.remedy}")
    failures = sum(1 for c in checks if c.status == FAIL)
    warnings = sum(1 for c in checks if c.status == WARN)
    lines.append("=" * (width + 70))
    lines.append(
        ("Ready to run" if not failures else "Fix the FAIL item(s) above first")
        + f"  ({failures} fail, {warnings} warn, {sys.platform})")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    network = "--network" in argv
    quiet = "--quiet" in argv
    checks = run_checks(network=network)
    if not quiet:
        print(format_report(checks))
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
