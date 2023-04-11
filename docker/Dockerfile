FROM amr-registry.caas.intel.com/aiops/transfer-learning:prod

COPY . /workspace

WORKDIR /workspace

RUN python -m pip install --no-cache-dir -r /workspace/frameworks.ai.transfer-learning/workflows/vision_anomaly_detection/requirements.txt

RUN python -m pip install --no-cache-dir -r /workspace/requirements.txt

ENV PYTHONPATH=/workspace