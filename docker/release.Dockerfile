# syntax=docker/dockerfile:1.7
# Build with the extracted Debian release archive as the context; no compiler is needed.
FROM debian:13-slim
ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_REQUIREMENT="cuda>=13.0"
COPY packages/ /tmp/ninfer-packages/
RUN apt-get update \
    && sort -u /tmp/ninfer-packages/*.txt | xargs apt-get install -y --no-install-recommends ca-certificates curl aria2 \
    && rm -rf /var/lib/apt/lists/* /tmp/ninfer-packages
COPY lib/ /opt/ninfer/lib/
COPY NVIDIA-CUDA-EULA.txt /opt/ninfer/NVIDIA-CUDA-EULA.txt
COPY --chmod=0755 bin/multi/ /opt/ninfer/
COPY --chmod=0755 run.sh download-model.sh /opt/ninfer/
ENV LD_LIBRARY_PATH=/opt/ninfer/lib \
    NVIDIA_VISIBLE_DEVICES=all \
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
