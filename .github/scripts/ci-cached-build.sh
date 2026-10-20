#!/usr/bin/env bash
# Keep the cache server outside the cancelled docker-exec process.
set -euo pipefail
container="ninfer-build-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
case "${1:?start|probe|build|finish}" in
  start)
    docker run -d --name "$container" --init -v "$PWD:/src" -w /src \
      -v "$SCCACHE_PATH:/usr/local/bin/sccache:ro" -v "$pythonLocation:$pythonLocation:ro" \
      -e SCCACHE_GHA_ENABLED=on -e SCCACHE_IDLE_TIMEOUT=0 \
      -e ACTIONS_CACHE_SERVICE_V2 -e ACTIONS_RESULTS_URL -e ACTIONS_RUNTIME_TOKEN \
      -e NINFER_COMPILER_LAUNCHER=sccache -e NINFER_CUDA_ARCH \
      -e "ARCHS=$NINFER_CUDA_ARCH" -e GITHUB_RUN_ID -e GITHUB_RUN_ATTEMPT \
      -e NINFER_MULTICALL=ON -e NINFER_BUILD_DIR=/src/build-ci \
      ninfer-toolchain:ci sleep infinity
    # Start the server before the build step, so cancelling that step leaves it alive.
    docker exec "$container" sccache --start-server
    ;;
  probe)
    docker exec "$container" "$pythonLocation/bin/python3.11" .github/scripts/check-compiler-cache.py
    ;;
  build)
    docker exec "$container" "$pythonLocation/bin/python3.11" .github/scripts/run-cached-build.py \
      --pid-file build-ci.pid --log build-ci.log -- \
      bash -c 'scripts/build-native.sh configure && scripts/build-native.sh build'
    ;;
  finish)
    if docker container inspect "$container" >/dev/null 2>&1; then
      # A cancellation of the host docker client need not stop the exec inside it.
      # Signal the recorded supervisor, then drain its independent cache server.
      trap 'docker rm -f "$container" >/dev/null' EXIT
      docker exec "$container" "$pythonLocation/bin/python3.11" .github/scripts/finish-compiler-cache.py \
        --pid-file build-ci.pid --stats build-ci-cache.json
    fi
    ;;
  *) exit 2 ;;
esac
