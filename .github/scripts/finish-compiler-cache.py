"""Stop new build work and allow sccache's bounded shutdown to finish."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from build_telemetry import sanitize_cache_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--stats", type=Path, required=True)
    args = parser.parse_args()
    if args.pid_file and args.pid_file.exists():
        try:
            os.kill(int(args.pid_file.read_text()), signal.SIGINT)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 6
        while args.pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.2)
    result = subprocess.run(["sccache", "--show-stats", "--stats-format=json"],
                            capture_output=True, text=True, timeout=20, check=True)
    stats = json.loads(result.stdout)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    subprocess.run(["sccache", "--stop-server"], check=True, timeout=20)
    # v0.18.0 acknowledges shutdown before its server finishes. The server allows
    # up to ten seconds for active requests and pending cache writes to finish.
    time.sleep(11)
    if os.environ.get("SCCACHE_ERROR_LOG"):
        sanitize_cache_log(Path(os.environ["SCCACHE_ERROR_LOG"]), args.stats.with_name("cache-errors.log"))
    if stats["stats"].get("cache_write_errors", 0):
        print("::warning::Some compiler cache writes failed; see the saved cache statistics.")


if __name__ == "__main__":
    main()
