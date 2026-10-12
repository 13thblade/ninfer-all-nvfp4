# CLAUDE.md

`AGENTS.md` holds the project rules and the Windows build recipe. This file adds what Claude sessions
on this machine need to share the GPU safely.

## GPU lock (one RTX 3090, many sessions)

Several sessions and agents often work on this machine at once, and they share a single RTX 3090.
Two processes on the card at the same time spoil every measurement and can fail for lack of VRAM. They
coordinate through one lock file:

```
%TEMP%\ninfer-gpu.lock        (C:\Users\Ash\AppData\Local\Temp\ninfer-gpu.lock)
```

The lock is **an open file handle with no sharing**, not the file's existence. The file stays on
disk permanently. Whoever holds a handle opened with `FileShare.None` owns the GPU, and every other
open fails with a sharing violation until that handle is closed. Windows closes the handle when its
process exits, so a crashed or killed holder can never leave the lock stuck.

### What needs the lock

Hold it for **any process that touches CUDA**:

- `ninfer-serve` / `run.bat` server starts, for the server's whole lifetime;
- benches, perplexity runs, `nsys`/`ncu` profiling, and Strata or any other GPU program;
- `ctest` runs. A regex that looks CPU-only can still pick up CUDA tests that have no `gpu` label,
  such as `ninfer_context_kv_materialize_test`. If unsure, take the lock.

These do **not** need it: compiling and linking (`cmake --build`), file copies, conversions that run
only on the CPU, and reading logs.

### How to take it (PowerShell)

Acquire it, run the GPU work, and release it, all inside **one** PowerShell invocation or script.
Shell state does not survive between tool calls, so a handle opened in one call is gone, and the lock
released, by the next.

```powershell
$lockPath = Join-Path $env:TEMP 'ninfer-gpu.lock'
$lock = $null
while (-not $lock) {
  try   { $lock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None') }
  catch [System.IO.IOException] { Start-Sleep -Milliseconds 500 }   # someone else holds it
}
try {
  # ... GPU work: bench, ctest, server run ...
} finally {
  $lock.Dispose()                                              # release promptly
}
```

- Waiting can take hours while another agent benchmarks. Put the loop in a script and run it with
  `run_in_background`, rather than polling from the conversation.
- Poll quickly, about every 0.5 s, as above. The lock has no queue: when it is released, the first
  waiter to retry gets it. A waiter polling every 15 s lost to faster pollers again and again and
  waited hours for a few minutes of tests.
- Keep each A/B (ABBA) set inside one hold, so another agent can't run between its halves.
  Release between phases so others can get a turn.
- For a server, keep the handle open until the server process exits, not just until it starts.

### Mistakes that have happened

- **Creating or deleting the file as the lock** (`FileMode.CreateNew`, `New-Item`, `Remove-Item`).
  Other sessions treat a present file as "free" and a missing file as "taken", or the reverse, so
  one agent stalled for about 5 hours waiting on a lock nobody held. Only ever open the existing
  file with `OpenOrCreate` and `FileShare.None`. Never create it fresh, never delete it.
- **Taking the lock in one tool call and running the GPU work in the next.** The handle closed in
  between, so the work ran unlocked.
- **Running `ctest` without the lock** because the regex looked CPU-only. See above.

### Checking whether the GPU is really in use

If a lock wait feels long, check whether the holder is actually working before suspecting a bug:

```powershell
try { $h = [IO.File]::Open("$env:TEMP\ninfer-gpu.lock", 'OpenOrCreate', 'ReadWrite', 'None'); $h.Dispose(); 'lock free' } catch { 'lock held' }
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
Get-Process ninfer*, nsys* -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime, CPU
```

- **Working:** 250-310 W and high utilization means the holder is busy. Flash-Next runs first spend
  about a minute near 40 W, reading the model into RAM.
- **Stuck:** a held lock with an idle GPU and no `ninfer*` process means something is wrong. Find it
  by command line (`Get-CimInstance Win32_Process`).

### The user's own processes

The user's own `ninfer-serve`, games and desktop apps do **not** take the lock. If the GPU is busy
while the lock is free, wait. Don't measure on a busy card, and don't stop the user's processes,
WSL or Docker. Report it if it blocks the work. `AGENTS.md` adds: ask before assuming the device is
free.

The protocol is Windows-only. Share modes don't carry over to WSL `flock`, so GPU work launched from
WSL needs the Windows-side lock held around it.
