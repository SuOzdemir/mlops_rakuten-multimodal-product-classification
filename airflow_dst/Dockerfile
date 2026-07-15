FROM apache/airflow:2.10.0

# Training runs in the ephemeral trainer container now (DockerOperator, via
# docker-socket-proxy) -- no torch/transformers/mlflow needed in this image.
#
# --constraint pins every package (including apache-airflow-providers-docker)
# to versions known to work with this exact apache-airflow==2.10.0 + Python
# combination. Without it, pip resolved providers-docker to a version that
# requires apache-airflow>=3, silently upgraded the base image's airflow
# 2.10.0 -> 3.3.0, and broke the `airflow` entrypoint entirely.
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.0/constraints-${PYTHON_VERSION%.*}.txt"
