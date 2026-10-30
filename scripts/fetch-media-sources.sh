#!/usr/bin/env bash
set -euo pipefail

# Downloads the FFmpeg and dav1d release archives that Linux builds compile and link statically
# (cmake/MediaSources.cmake pins them by SHA-256). The build itself never downloads anything, so
# offline and hermetic builds stay that way: run this first, once per checkout.
#
#   fetch-media-sources.sh [dir]
#
# dir defaults to third_party/media-sources (ignored by git), where CMake looks unless configured
# with -DNINFER_MEDIA_SOURCES_DIR=<dir>.

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dir=${1:-$root/third_party/media-sources}
cmake -DNINFER_MEDIA_SOURCES_DIR="$dir" -P "$root/cmake/FetchMediaSources.cmake"
