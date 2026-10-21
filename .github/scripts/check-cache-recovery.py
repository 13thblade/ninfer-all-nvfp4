"""Compile one object to check cache transport between interrupted GitHub jobs."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from build_telemetry import cache_stats

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("seed", "verify", "wait"))
args = parser.parse_args()
if args.mode == "wait":
    while True:
        print("Completed object is cached; waiting for timeout or cancellation.", flush=True)
        time.sleep(15)

identity = os.environ["CACHE_SOURCE_RUN"]
if not identity.isdecimal():
    raise ValueError("CACHE_SOURCE_RUN must be a workflow run ID")
root = Path("build-cache-recovery")
root.mkdir(exist_ok=True)
(root / "CMakeLists.txt").write_text(
    "cmake_minimum_required(VERSION 3.28)\nproject(recovery LANGUAGES CXX)\n"
    "set(CMAKE_CXX_SCAN_FOR_MODULES OFF)\nadd_library(recovery OBJECT recovery.cpp)\n")
(root / "recovery.cpp").write_text(f"int recovery_{identity}() {{ return 42; }}\n")
subprocess.run(["cmake", "-S", str(root), "-B", str(root / "out"), "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"], check=True)
subprocess.run(["ccache", "--zero-stats"], check=True)
subprocess.run(["cmake", "--build", str(root / "out"), "--clean-first", "-j"], check=True)
stats = cache_stats()
hits = stats["direct_cache_hit"] + stats["preprocessed_cache_hit"]
misses = stats["cache_miss"]
print(f"Recovery {args.mode}: {hits} hits, {misses} misses", flush=True)
expected = (1, 0) if args.mode == "verify" else (0, 1)
if (hits, misses) != expected:
    raise SystemExit(f"Expected hits/misses {expected}; cache recovery did not pass")
