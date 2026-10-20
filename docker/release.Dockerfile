# syntax=docker/dockerfile:1.7
# Build with the extracted Debian release archive as the context; no compiler is needed.
FROM debian:13-slim
ARG CUDA_REPO=debian13
ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_REQUIREMENT="cuda>=13.0"
COPY packages/ /tmp/ninfer-packages/
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSLo /tmp/cuda-keyring.deb \
       https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && apt-get update \
    && sort -u /tmp/ninfer-packages/*.txt | xargs apt-get install -y --no-install-recommends aria2 \
    && find /usr/local -name 'libcublas.so.*' -printf '%h\n' | sort -u > /etc/ld.so.conf.d/ninfer-cuda.conf \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/* /tmp/cuda-keyring.deb /tmp/ninfer-packages
COPY --chmod=0755 bin/multi/ /opt/ninfer/
COPY --chmod=0755 run.sh download-model.sh /opt/ninfer/
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA=${CUDA_REQUIREMENT} \
    NINFER_MODEL_DIR=/models \
    NINFER_DEVICE_PROFILES=/cache/device-profiles.json \
    NINFER_HOST=0.0.0.0 \
    NINFER_PORT=8080
VOLUME ["/models", "/grafts", "/cache"]
WORKDIR /models
EXPOSE 8080
STOPSIGNAL SIGTERM
ENTRYPOINT ["/opt/ninfer/run.sh"]
CMD ["qwen38-27b"]
