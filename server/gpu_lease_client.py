"""Client for an OPTIONAL external GPU lease script.

Why this exists: a small GPU cannot host the embedding model and another GPU job
(transcription, OCR, a vision model) at the same time. When both run, the Ollama
runner dies under VRAM pressure and whole embedding batches fail — which shows up
as silently dropped chunks, not as an error you would notice.

If you have a machine-wide lease script, point VAULT_SEARCH_GPU_LEASE at it and
the indexers will take a turn like any other job. The script is expected to
accept:

    <script> acquire --name NAME --timeout SECONDS
    <script> release --name NAME
    <script> heartbeat --name NAME
    <script> waiters                 # exit 0 = someone is queued, 1 = nobody

**Nothing here is required.** With no script configured (the default), every
method is a no-op and indexing simply runs — which is the right behavior on a
machine that has the GPU to itself.

Two policies are deliberate, and both were learned the hard way:

  1. ALWAYS acquire before touching the GPU, and refuse to run if the lease
     cannot be had. The older "only acquire if another job is currently
     RUNNING" check made the indexer an unregistered squatter whenever it
     started in the gap right after another job released, starving every job
     queued behind it.
  2. YIELD between files. A full index run holds the card for hours, while the
     jobs queued behind it typically need a minute per file. Without yielding,
     a short job can wait most of a day.
"""

import subprocess
import sys
import time
from pathlib import Path

# How long a `waiters` probe may be skipped when we have no queue directory to
# peek at. Bounds the subprocess cost when yield_wanted() is called per file.
WAITERS_MIN_INTERVAL_S = 30.0


class GpuLease:
    """A turn at the GPU, taken from an external lease script.

    Every method is safe to call when no script is configured; the lease is
    simply never held and the caller runs unsynchronized.
    """

    def __init__(self, name, *, script=None, queue_dir=None,
                 acquire_timeout=21600.0, min_hold_s=180.0, runner=None):
        self.name = name
        self.script = Path(script) if script else None
        self.queue_dir = Path(queue_dir) if queue_dir else None
        self.acquire_timeout = float(acquire_timeout)
        self.min_hold_s = float(min_hold_s)
        self._runner = runner or subprocess.run
        self._held = False
        self._held_since = 0.0
        self._last_waiters_check = 0.0

    # --- state ---------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True when a lease script is configured AND present on disk."""
        return bool(self.script) and self.script.exists()

    @property
    def held(self) -> bool:
        return self._held

    # --- plumbing ------------------------------------------------------------

    def _run(self, *cli, timeout=60.0) -> tuple[bool, str]:
        """Run the lease script. Returns (ok, stdout+stderr)."""
        if not self.enabled:
            return False, "no lease script configured"
        try:
            r = self._runner(
                [sys.executable, str(self.script), *cli],
                capture_output=True, text=True, timeout=timeout,
            )
            return r.returncode == 0, (r.stdout or "") + (r.stderr or "")
        except Exception as e:
            return False, f"error: {e}"

    # --- lifecycle -----------------------------------------------------------

    def acquire(self) -> tuple[bool, str]:
        """Take a turn. Returns (ok, detail).

        NOT best-effort: a False here means someone else holds the card, and
        the caller must abort rather than run unregistered.
        With no script configured this returns (True, ...) and holds nothing.
        """
        if not self.enabled:
            return True, "no GPU lease configured — running unsynchronized"
        ok, out = self._run(
            "acquire", "--name", self.name, "--timeout", str(int(self.acquire_timeout)),
            # The script itself blocks until the timeout, so give the subprocess
            # room beyond it rather than killing a wait that was about to win.
            timeout=self.acquire_timeout + 300,
        )
        if ok:
            self._held = True
            self._held_since = time.time()
        return ok, out

    def release(self) -> tuple[bool, str]:
        if not self._held:
            return True, ""
        ok, out = self._run("release", "--name", self.name)
        self._held = False
        return ok, out

    def heartbeat(self) -> None:
        """Refresh the holder timestamp so a stale-lease reaper leaves us alone.

        A full run can hold the lease for many hours — far longer than the
        window a reaper uses to decide a holder has died.
        """
        if self._held:
            self._run("heartbeat", "--name", self.name)

    # --- cooperative yielding ------------------------------------------------

    def yield_wanted(self) -> bool:
        """True when a job is queued behind us and we have held long enough.

        Cheap filesystem peek first when we know the queue directory, then the
        authoritative `waiters` call — that one reaps dead tickets, so a killed
        waiter cannot make us thrash release/re-acquire forever. Without a queue
        directory the probe is time-throttled instead, since this is called once
        per file.
        """
        if not self._held:
            return False
        if time.time() - self._held_since < self.min_hold_s:
            return False
        if self.queue_dir is not None:
            try:
                if not any(self.queue_dir.iterdir()):
                    return False
            except OSError:
                return False
        else:
            now = time.time()
            if now - self._last_waiters_check < WAITERS_MIN_INTERVAL_S:
                return False
            self._last_waiters_check = now
        ok, _ = self._run("waiters")
        return ok  # non-zero exit = queue empty (or only dead tickets)

    def yield_now(self, unload_model: str | None = None) -> None:
        """Hand the card over, then re-acquire at the back of the queue.

        The caller MUST have flushed every pending embed batch, DB write and
        cache before calling this: once we release, another process owns the
        GPU, and if the re-acquire fails we abort.
        """
        if not self._held:
            return
        # Releasing the lease does not evict the model from VRAM, and the next
        # job's free-memory check will fail while it is still resident.
        if unload_model:
            try:
                self._runner(["ollama", "stop", unload_model],
                             capture_output=True, timeout=60)
            except Exception:
                pass
        print("[gpu-lease] yielding to queued job(s) — release + re-acquire", flush=True)
        self.release()
        ok, out = self.acquire()
        if not ok:
            print(f"[gpu-lease] re-acquire after yield FAILED — aborting cleanly. "
                  f"{out.strip()[-200:]}", flush=True)
            raise SystemExit(2)
        print("[gpu-lease] resumed after yield", flush=True)


def acquire_or_exit(lease: GpuLease) -> None:
    """Take the lease before any GPU work, or stop.

    Running without it would make this process an unregistered squatter that
    collides with whatever does hold the card — and the collision costs dropped
    chunks, silently.
    """
    ok, detail = lease.acquire()
    if not ok:
        print(f"[gpu-lease] acquire FAILED — refusing to run without the lease. "
              f"{detail.strip()[-200:]}", flush=True)
        sys.exit(2)
    if lease.held:
        print(f"[gpu-lease] acquired as '{lease.name}'", flush=True)
    else:
        print(f"[gpu-lease] {detail}", flush=True)
