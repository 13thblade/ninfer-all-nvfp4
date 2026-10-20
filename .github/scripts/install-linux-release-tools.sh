#!/usr/bin/env bash
# Build on Debian 13; install the Toolkit without a GPU driver.
set -euo pipefail
: "${CUDA_VERSION:?}"
minor="${CUDA_VERSION%.*}"
suffix="${minor/./-}"
cuda_repo=debian13
[[ "$CUDA_VERSION" == 12.* ]] && cuda_repo=debian12
apt-get update
apt-get install -y --no-install-recommends \
  build-essential ninja-build pkg-config python3 python3-venv \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev \
  libcurl4-openssl-dev zlib1g-dev
curl -fsSLo /tmp/cuda-keyring.deb \
  "https://developer.download.nvidia.com/compute/cuda/repos/$cuda_repo/x86_64/cuda-keyring_1.1-1_all.deb"
dpkg -i /tmp/cuda-keyring.deb
apt-get update
apt-get install -y --no-install-recommends "cuda-toolkit-$suffix=$CUDA_VERSION-1"
python3 -m venv /opt/ninfer-bootstrap
/opt/ninfer-bootstrap/bin/pip install --disable-pip-version-check uv==0.11.11
/opt/ninfer-bootstrap/bin/uv python install 3.11
/opt/ninfer-bootstrap/bin/uv venv --python 3.11 --seed /opt/ninfer-build-tools
/opt/ninfer-build-tools/bin/pip install --disable-pip-version-check cmake==3.31.10
echo '/opt/ninfer-build-tools/bin' >> "$GITHUB_PATH"
echo "/usr/local/cuda-$minor/bin" >> "$GITHUB_PATH"
echo "CUDACXX=/usr/local/cuda-$minor/bin/nvcc" >> "$GITHUB_ENV"
echo "CUDA_PATH=/usr/local/cuda-$minor" >> "$GITHUB_ENV"
