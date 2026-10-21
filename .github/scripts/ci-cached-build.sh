#!/usr/bin/env bash
# Keep completed compiler outputs on the workspace mount across cancellation.
set -euo pipefail
container="ninfer-build-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
case "${1:?start|probe|build|finish}" in
  start)
    docker run -d --name "$container" --init -v "$PWD:/src" -w /src \
      -v "$CCACHE_PATH:/usr/local/bin/ccache:ro" -v "$pythonLocation:$pythonLocation:ro" \
      -e CCACHE_DIR=/src/.cache/objects -e CCACHE_MAXSIZE -e CCACHE_COMPILERCHECK -e CCACHE_DEPEND -e CCACHE_COMPRESSLEVEL \
      -e NINFER_COMPILER_LAUNCHER=ccache -e NINFER_CUDA_ARCH \
      -e "ARCHS=$NINFER_CUDA_ARCH" -e GITHUB_RUN_ID -e GITHUB_RUN_ATTEMPT \
      -e NINFER_MULTICALL=ON -e NINFER_BUILD_DIR=/src/build-ci \
      ninfer-toolchain:ci sleep infinity
    ;;
  probe)
    docker exec "$container" "$pythonLocation/bin/python3.11" .github/scripts/check-compiler-cache.py
    ;;
  build)
    docker exec "$container" "$pythonLocation/bin/python3.11" .github/scripts/run-cached-build.py \
      --pid-file build-ci.pid --log build-ci.log --cache-stats build-ci-cache-progress.jsonl -- \
      bash -c 'scripts/build-native.sh configure && scripts/build-native.sh build'
    ;;
  finish)
    if docker container inspect "$container" >/dev/null 2>&1; then
      # A cancellation of the host docker client need not stop the exec inside it.
      # Signal the recorded supervisor before saving completed cache entries.
      trap 'docker rm -f "$container" >/dev/null' EXIT
      docker exec "$container" "$pythonLocation/bin/python3.11" .github/scripts/finish-compiler-cache.py \
        --pid-file build-ci.pid --stats build-ci-cache.json
    fi
    ;;
  *) exit 2 ;;
esac
