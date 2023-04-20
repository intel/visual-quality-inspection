FROM amr-registry.caas.intel.com/aiops/transfer-learning:prod-r0.4

RUN apt-get update -y && apt-get install -y --fix-missing --no-install-recommends wget

COPY . /workspace

WORKDIR /workspace

RUN python -m pip install --no-cache-dir -r /workspace/intel-models/datasets/dataset_api/requirements.txt

RUN python -m pip install --no-cache-dir -r /workspace/transfer-learning/workflows/vision_anomaly_detection/requirements.txt

RUN python -m pip install --no-cache-dir -r /workspace/requirements.txt

ENV PYTHONPATH=/workspace
