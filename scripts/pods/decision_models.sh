#!/usr/bin/env bash
# The artifact the decision checks serve: the published Ternary Bonsai 2 27B, a Qwen3.5-core model.
set -Eeuo pipefail
root=/workspace/ninfer-work
mkdir -p "$root/models"
[ -f "$root/models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer" ] || \
    "$root/py311/bin/hf" download WaveCut/Ternary-Bonsai-2-27B-NInfer-v3 \
        Ternary-Bonsai-2-27B-ninfer-v3.ninfer --local-dir "$root/models"
ls -l "$root/models/Ternary-Bonsai-2-27B-ninfer-v3.ninfer"
