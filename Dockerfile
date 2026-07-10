ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-venv git && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY . .
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install -e .
ENV PATH="/opt/venv/bin:${PATH}"
ENTRYPOINT ["bash"]
