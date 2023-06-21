ARG IMAGE_NAME=intel/ai-tools
ARG IMAGE_TAG=tlt-0.5.0
FROM ${IMAGE_NAME}:${IMAGE_TAG}

RUN apt-get update -y && apt-get install -y --fix-missing --no-install-recommends wget xz-utils

COPY . /workspace

WORKDIR /workspace

RUN python -m pip install --no-cache-dir dataset-librarian

RUN python -m pip install --no-cache-dir -r /workspace/transfer-learning/workflows/vision_anomaly_detection/requirements.txt

RUN python -m pip install --no-cache-dir -r /workspace/requirements.txt

ENV PYTHONPATH=/workspace
