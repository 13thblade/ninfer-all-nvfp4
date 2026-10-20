#!/usr/bin/env python3
"""Check that completed C++ and CUDA compilations survive a cache-server restart."""
import json
import os
from pathlib import Path
import subprocess
import sys


def run(*args):
    subprocess.run(args, check=True)


root = Path("build-cache-check")
root.mkdir(exist_ok=True)
(root / "CMakeLists.txt").write_text("""
cmake_minimum_required(VERSION 3.28)
project(cache_check LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
add_library(cache_check OBJECT host.cpp kernel.cu)
target_compile_options(cache_check PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;-Xfatbin=-compress-all;-compress-mode=size>")
""")
# A new run must write its own entries before the second build can read them.
identity = os.environ["GITHUB_RUN_ID"] + "_" + os.environ["GITHUB_RUN_ATTEMPT"]
(root / "host.cpp").write_text(f'int host_{identity}() {{ return 42; }}\n')
(root / "kernel.cu").write_text(
    f'__global__ void kernel_{identity}(int *p) {{ p[threadIdx.x] = 42; }}\n')
run("cmake", "-S", str(root), "-B", str(root / "out"), "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=" + os.environ["ARCHS"],
    "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache", "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache")
run("sccache", "--zero-stats")
run("cmake", "--build", str(root / "out"), "-j")
run("sccache", "--stop-server")  # Flush writes; no process-local state can serve the next build.
run("cmake", "--build", str(root / "out"), "--clean-first", "-j")
result = subprocess.check_output(["sccache", "--show-stats", "--stats-format=json"], text=True)
(root / "cache-stats.json").write_text(result)
data = json.loads(result)
stats = data["stats"]
hits = sum(stats["cache_hits"]["counts"].values())
misses = sum(stats["cache_misses"]["counts"].values())
print(f"After server restart: {hits} cache hits, {misses} misses", flush=True)
run("sccache", "--show-stats")
if hits < 2 or misses or stats["cache_write_errors"] or stats["requests_not_cacheable"]:
    sys.exit("C++/CUDA cache reuse failed; do not start the long build")
run("sccache", "--zero-stats")
