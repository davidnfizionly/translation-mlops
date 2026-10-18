# Tooling image to define, upsert and start the SageMaker Pipeline and jobs.
# The SageMaker Python SDK is pinned to the 2.x line (stable Pipelines API).
FROM python:3.12-slim
RUN pip install --no-cache-dir --root-user-action=ignore "sagemaker==2.257.7" "pytest==8.*"
WORKDIR /repo
