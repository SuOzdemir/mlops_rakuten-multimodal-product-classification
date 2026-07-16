"""Run model registration, gating, and deployment in an ephemeral container."""

import os
from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

HOST_PROJECT_DIR = os.environ["HOST_PROJECT_DIR"]
PROMOTION_IMAGE = os.environ.get(
    "PROMOTION_IMAGE", "mlops-rakuten-product-classification-promotion:latest"
)
PROMOTION_NETWORK = os.environ.get(
    "PROMOTION_NETWORK", "mlops-rakuten-product-classification_default"
)
DOCKER_PROXY_URL = os.environ.get("DOCKER_PROXY_URL", "tcp://docker-socket-proxy:2375")

with DAG(
    dag_id="rakuten_model_promotion",
    description="Gate and register a candidate model in an ephemeral promotion container.",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "registry", "deployment", "rakuten"],
) as dag:
    register_gate_and_deploy = DockerOperator(
        task_id="register_gate_and_deploy",
        image=PROMOTION_IMAGE,
        docker_url=DOCKER_PROXY_URL,
        network_mode=PROMOTION_NETWORK,
        auto_remove="success",
        mount_tmp_dir=False,
        mounts=[Mount(source=HOST_PROJECT_DIR, target="/project", type="bind")],
        mem_limit="8g",
        shm_size="2gb",
        environment={
            "MODEL": "{{ dag_run.conf.get('model', '') }}",
            "RAKUTEN_PROJECT_DIR": "/project",
            "CAMEMBERT_BASE_DIR": os.environ.get(
                "CAMEMBERT_BASE_DIR",
                "/project/data/rakuten_streamlit_predictor/text_model/camembert_run4",
            ),
            "MLFLOW_TRACKING_URI": os.environ.get(
                "MLFLOW_TRACKING_URI", "http://mlflow:5001"
            ),
            "DVC_ENDPOINT_URL": os.environ.get("DVC_ENDPOINT_URL", "http://minio:9000"),
            "PROMOTION_MIN_F1_GAIN": os.environ.get("PROMOTION_MIN_F1_GAIN", "0"),
            "PROMOTION_ALLOW_UNTAGGED_CHAMPION": os.environ.get(
                "PROMOTION_ALLOW_UNTAGGED_CHAMPION", "false"
            ),
        },
    )
