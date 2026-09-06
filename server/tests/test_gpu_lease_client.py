"""Tests for the optional GPU lease client.

The lease exists to stop two GPU jobs sharing a card that cannot hold both. When
it misbehaves the symptom is never an error — it is silently dropped chunks, or a
one-minute job waiting behind a six-hour one. So the properties worth pinning are
the policy decisions, not the plumbing:

  - No script configured (the DEFAULT for this project) must be a clean no-op.
    Every method has to stay safe to call, and indexing must still run.
  - A failed acquire must stop the run. Continuing unregistered is what made the
    old implementation a squatter.
  - Yielding must respect the minimum hold, or a burst of queued jobs turns the
    indexer into a release/re-acquire thrash that never indexes anything.
  - A failed re-acquire after yielding must abort, not silently continue without
    the card.

Plain stdlib with an injected runner — no subprocess is ever spawned.

Run directly (`python server/tests/test_gpu_lease_client.py`) or under pytest.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpu_lease_client import GpuLease, acquire_or_exit  # noqa: E402


class FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeRunner:
    """Records calls; answers each subcommand from `results`."""

    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        # argv is [python, script, subcommand, ...] or ["ollama", "stop", model]
        if argv[0] == "ollama":
            return FakeProc(0)
        sub = argv[2] if len(argv) > 2 else ""
        return self.results.get(sub, FakeProc(0))

    def subcommands(self):
        return [c[2] for c in self.calls if c[0] != "ollama" and len(c) > 2]


def _script(tmp: Path) -> Path:
    p = tmp / "gpu_lease.py"
    p.write_text("# stand-in for a real lease script\n", encoding="utf-8")
    return p


# --- the default: no lease script -------------------------------------------

def test_no_script_configured_is_a_clean_noop():
    """The default install has no lease script and must simply run."""
    runner = FakeRunner()
    lease = GpuLease("job", script=None, runner=runner)

    assert lease.enabled is False
    ok, detail = lease.acquire()

    assert ok is True          # caller proceeds...
    assert lease.held is False  # ...but nothing is actually held
    assert "no GPU lease" in detail
    # Nothing may be called on a lease that does not exist
    lease.heartbeat()
    assert lease.yield_wanted() is False
    lease.yield_now(unload_model="some-model")
    assert lease.release() == (True, "")
    assert runner.calls == []


def test_configured_path_that_does_not_exist_is_also_a_noop():
    """A stale path in the config must not break indexing."""
    with tempfile.TemporaryDirectory() as d:
        lease = GpuLease("job", script=Path(d) / "not-here.py", runner=FakeRunner())
        assert lease.enabled is False
        assert lease.acquire()[0] is True
        assert lease.held is False


# --- acquire / release ------------------------------------------------------

def test_acquire_registers_under_the_configured_name():
    with tempfile.TemporaryDirectory() as d:
        runner = FakeRunner()
        lease = GpuLease("vault_index", script=_script(Path(d)), runner=runner)

        ok, _ = lease.acquire()

        assert ok is True and lease.held is True
        argv = runner.calls[0]
        assert argv[2] == "acquire"
        assert "--name" in argv and "vault_index" in argv


def test_failed_acquire_does_not_report_the_lease_as_held():
    with tempfile.TemporaryDirectory() as d:
        runner = FakeRunner({"acquire": FakeProc(1, stderr="timeout")})
        lease = GpuLease("job", script=_script(Path(d)), runner=runner)

        ok, out = lease.acquire()

        assert ok is False
        assert lease.held is False
        assert "timeout" in out


def test_acquire_or_exit_stops_the_run_when_the_lease_cannot_be_had():
    """Continuing here is what starved queued jobs in the old implementation."""
    with tempfile.TemporaryDirectory() as d:
        runner = FakeRunner({"acquire": FakeProc(1, stderr="held by another job")})
        lease = GpuLease("job", script=_script(Path(d)), runner=runner)

        try:
            acquire_or_exit(lease)
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError("acquire_or_exit must exit when acquire fails")


def test_acquire_or_exit_proceeds_when_no_script_is_configured():
    acquire_or_exit(GpuLease("job", script=None, runner=FakeRunner()))  # must not raise


def test_release_is_not_sent_when_nothing_is_held():
    with tempfile.TemporaryDirectory() as d:
        runner = FakeRunner()
        lease = GpuLease("job", script=_script(Path(d)), runner=runner)
        assert lease.release() == (True, "")
        assert runner.calls == []


# --- yielding ---------------------------------------------------------------

def test_yield_is_refused_before_the_minimum_hold():
    """Otherwise a queue of short jobs makes the indexer thrash."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        queue = tmp / "queue"
        queue.mkdir()
        (queue / "ticket-1").write_text("", encoding="utf-8")
        lease = GpuLease("job", script=_script(tmp), queue_dir=queue,
                         min_hold_s=3600, runner=FakeRunner())
        lease.acquire()

        assert lease.yield_wanted() is False


def test_yield_wanted_only_when_something_is_actually_queued():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        queue = tmp / "queue"
        queue.mkdir()
        runner = FakeRunner()
        lease = GpuLease("job", script=_script(tmp), queue_dir=queue,
                         min_hold_s=0, runner=runner)
        lease.acquire()

        # Empty queue: answered by the directory peek, no subprocess at all
        assert lease.yield_wanted() is False
        assert "waiters" not in runner.subcommands()

        (queue / "ticket-1").write_text("", encoding="utf-8")
        assert lease.yield_wanted() is True
        assert "waiters" in runner.subcommands()


def test_dead_tickets_do_not_trigger_a_yield():
    """`waiters` reaps dead tickets; a non-zero exit means nobody real is waiting."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        queue = tmp / "queue"
        queue.mkdir()
        (queue / "ticket-dead").write_text("", encoding="utf-8")
        lease = GpuLease("job", script=_script(tmp), queue_dir=queue, min_hold_s=0,
                         runner=FakeRunner({"waiters": FakeProc(1)}))
        lease.acquire()

        assert lease.yield_wanted() is False


def test_yield_now_releases_reacquires_and_unloads_the_model():
    with tempfile.TemporaryDirectory() as d:
        runner = FakeRunner()
        lease = GpuLease("job", script=_script(Path(d)), min_hold_s=0, runner=runner)
        lease.acquire()

        lease.yield_now(unload_model="embed-model")

        assert lease.held is True  # back at the end of the queue, but holding
        assert runner.subcommands() == ["acquire", "release", "acquire"]
        assert ["ollama", "stop", "embed-model"] in runner.calls


def test_failed_reacquire_after_yield_aborts():
    """Carrying on here would run the rest of the index without the card."""
    with tempfile.TemporaryDirectory() as d:
        calls = {"n": 0}

        class Runner(FakeRunner):
            def __call__(self, argv, **kwargs):
                super().__call__(argv, **kwargs)
                if len(argv) > 2 and argv[2] == "acquire":
                    calls["n"] += 1
                    if calls["n"] > 1:  # the re-acquire fails
                        return FakeProc(1, stderr="lost the queue")
                return FakeProc(0)

        lease = GpuLease("job", script=_script(Path(d)), min_hold_s=0, runner=Runner())
        lease.acquire()

        try:
            lease.yield_now()
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError("a failed re-acquire must abort the run")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"OK: {len(fns)} tests passed")
