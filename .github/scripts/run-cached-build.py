"""Run a build with bounded cancellation and periodic log progress."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from build_telemetry import sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--cache-stats", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a build command is required")
    cancelled = False

    def cancel(_signum, _frame):
        nonlocal cancelled
        cancelled = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, cancel)
    if os.name == "nt":
        signal.signal(signal.SIGBREAK, cancel)
    if args.pid_file:
        args.pid_file.write_text(str(os.getpid()))
    args.log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.log.open("wb") as log:
            options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                       if os.name == "nt" else {"start_new_session": True})
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **options)
            next_report = 0
            next_sample = 0
            while child.poll() is None and not cancelled:
                if args.cache_stats and time.monotonic() >= next_sample:
                    sample(args.cache_stats)
                    next_sample = time.monotonic() + 60
                if time.monotonic() >= next_report:
                    print(f"Build PID {child.pid}; log bytes: {args.log.stat().st_size}", flush=True)
                    with args.log.open("rb") as stream:
                        stream.seek(max(0, args.log.stat().st_size - 2048))
                        print(stream.read().decode(errors="replace").splitlines()[-2:], flush=True)
                    next_report = time.monotonic() + 15
                time.sleep(0.2)
            if cancelled:
                print("Stopping compiler work; the following cleanup step drains sccache.", flush=True)
                try:
                    if os.name == "nt":
                        child.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        os.killpg(child.pid, signal.SIGINT)
                    child.wait(timeout=4)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    if child.poll() is None:
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                           timeout=2, check=False)
                        else:
                            os.killpg(child.pid, signal.SIGKILL)
                return 130
            return child.returncode
    finally:
        if args.pid_file:
            args.pid_file.unlink(missing_ok=True)
        if args.log.exists():
            with args.log.open("rb") as stream:
                stream.seek(max(0, args.log.stat().st_size - 6000))
                print(stream.read().decode(errors="replace"), flush=True)


if __name__ == "__main__":
    sys.exit(main())
