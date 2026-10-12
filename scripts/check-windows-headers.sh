#!/usr/bin/env bash
# Syntax-checks every host C++ source against the Windows headers, with MinGW-w64 standing in for
# MSVC, so a Windows build break shows up on a Linux or macOS machine. It catches what the Windows
# headers do to portable code: <windows.h>, which nvtx3 brings into most translation units, defines
# `near`, `far` and other common words as macros, and a strict C++20 <cmath> declares no `M_PI`
# (issue #44). MinGW is GCC, so it does not see MSVC's own gaps (GCC builtins, `sincosf`, scalar
# F16C intrinsics); review covers those.
#
# usage: CUDA_INCLUDE_DIRS=DIR[:DIR...] scripts/check-windows-headers.sh [INCLUDE_DIR...]
#
# CUDA_INCLUDE_DIRS names the CUDA headers: cuda_runtime.h, crt/, cuda/ (CCCL), nvtx3/ and
# cuda_profiler_api.h; the include directories of the nvidia-cuda-runtime, -cuda-cccl, -cuda-nvcc,
# -nvtx and -cuda-profiler-api wheels serve. Further arguments are system include directories for
# FFmpeg, libcurl and zlib, which the media sources include. MINGW_CXX overrides the compiler
# (x86_64-w64-mingw32-g++-posix when installed, else x86_64-w64-mingw32-g++).
set -euo pipefail

root="$(git -C "$(dirname -- "$0")" rev-parse --show-toplevel)"
cd "$root"
if [[ -z "${MINGW_CXX:-}" ]]; then
  MINGW_CXX=x86_64-w64-mingw32-g++
  if command -v x86_64-w64-mingw32-g++-posix >/dev/null; then
    MINGW_CXX=x86_64-w64-mingw32-g++-posix
  fi
fi
command -v "$MINGW_CXX" >/dev/null || { echo "missing $MINGW_CXX" >&2; exit 2; }
[[ -n "${CUDA_INCLUDE_DIRS:-}" ]] || { echo "set CUDA_INCLUDE_DIRS" >&2; exit 2; }

work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT
mkdir "$work/generated" "$work/failed"
cmake -DSRC_DIR="$root" -DOUT="$work/generated/ninfer_build_id.h" \
  -P cmake/GenerateBuildId.cmake >/dev/null
# nvtx3 includes <Windows.h>; Windows file systems ignore the case, MinGW's headers on Linux do not.
printf '#include_next <windows.h>\n' >"$work/generated/Windows.h"

flags=(-std=c++20 -fsyntax-only -w
  # The definitions every Windows build compiles with (CMakeLists.txt, src/media).
  -DNOMINMAX -DWIN32_LEAN_AND_MEAN -DNINFER_MEDIA_NATIVE_PNG -DNINFER_TMA_STAGED_DESCRIPTORS=1
  -DNINFER_SM8X_COMPAT=1 -DUTF8PROC_STATIC "-DNINFER_SOURCE_DIR=\"$root\""
  -Iinclude -Isrc -Itests -Ibench -Iapps/cli -Iapps/perplexity -Iapps/serve -Ibench/inference
  -Ibench/context_cost -Ithird_party -Ithird_party/spdlog/include -Ithird_party/cpp-httplib
  -Ithird_party/xgrammar/include -Ithird_party/xgrammar/3rdparty/dlpack/include
  -Ithird_party/xgrammar/3rdparty/picojson -Ithird_party/utf8proc -Ithird_party/llama-jinja
  -isystem "$work/generated")
IFS=: read -r -a cuda_dirs <<<"$CUDA_INCLUDE_DIRS"
for dir in "${cuda_dirs[@]}" "$@"; do flags+=(-isystem "$dir"); done
# A GCC response file: one quoted argument per line.
for flag in "${flags[@]}"; do
  flag="${flag//\\/\\\\}"
  printf '"%s"\n' "${flag//\"/\\\"}"
done >"$work/flags"

# Sources no Windows build compiles: Linux preload shims and a probe built by a pod script.
git ls-files 'src/*.cpp' 'apps/*.cpp' 'tests/*.cpp' 'tools/*.cpp' 'bench/*.cpp' |
  grep -v -E '^(tests/models/qwen4_exp/delayed_[a-z_]+|bench/ops/q2_cpu_probe)\.cpp$' \
    >"$work/sources"
count="$(wc -l <"$work/sources" | tr -d ' ')"
jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
tr '\n' '\0' <"$work/sources" | xargs -0 -P "$jobs" -I{} sh -c '
  out=$("$1" @"$2" "$3" 2>&1) ||
    printf "%s\n" "$out" | grep -E "error" | head -n 8 >"$4/$(printf "%s" "$3" | tr / _)"
' _ "$MINGW_CXX" "$work/flags" {} "$work/failed"

failures="$(find "$work/failed" -type f | wc -l | tr -d ' ')"
if [[ "$failures" != 0 ]]; then
  cat "$work/failed"/* >&2
  echo "$failures of $count host sources do not compile against the Windows headers" >&2
  exit 1
fi
echo "$count host sources compile against the Windows headers"
