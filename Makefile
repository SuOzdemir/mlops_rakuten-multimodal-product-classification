.PHONY: install setup-data prepare-splits up serve up-all down down-all logs health airflow-up airflow-down airflow-logs test manual-up manual-down manual-up-all manual-down-all

# Local dev/training environment (pyproject.toml): notebooks, training scripts
install:
	uv sync

# Downloads the raw Kaggle dataset into data/raw/ (needs ~/.kaggle/kaggle.json)
setup-data:
	./scripts/setup_data.sh

# DVC prepare_splits stage, run locally (no Docker/Airflow needed)
prepare-splits:
	uv run dvc repro prepare_splits

# Serving stack: postgres -> mlflow -> api -> streamlit (order handled by depends_on/healthcheck)
# --wait blocks until all healthchecks pass, so `up-all`'s `up` then `airflow-up`
# ordering is safe (Airflow's DB needs postgres to actually be ready, not just started).
up:
	docker compose up --build -d --wait

# Everything: serving stack (mlflow+api+streamlit) + Airflow (own compose stack/network)
up-all: up airflow-up

down-all: down airflow-down

# Just api (:8000) + streamlit (:8501), no mlflow — same ports every time
serve:
	docker compose up --build -d api streamlit

down:
	docker compose down

logs:
	docker compose logs -f

health:
	curl -sf http://localhost:5001/health && echo " mlflow ok"
	curl -sf http://localhost:8000/health && echo " api ok"
	curl -sf http://localhost:8501/_stcore/health && echo " streamlit ok"

# Training/promotion stack: separate compose project, own network
airflow-up:
	cd airflow_dst && docker compose up --build -d

airflow-down:
	cd airflow_dst && docker compose down

airflow-logs:
	cd airflow_dst && docker compose logs -f

test:
	uv run pytest

# Same serving stack as `up`, but without Docker: mlflow + api + streamlit as
# background local processes (uv run), logs go to *.log (gitignored).
manual-up:
	PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns/artifacts > mlflow.log 2>&1 &
	MLFLOW_TRACKING_URI=http://localhost:5001 AIRFLOW_API_URL=http://localhost:8080 uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
	API_URL=http://localhost:8000 uv run streamlit run streamlit_app/Home.py --server.port 8501 > streamlit.log 2>&1 &

# Kills the background processes started by manual-up (matched by command line)
manual-down:
	-pkill -f "mlflow server"
	-pkill -f "uvicorn src.api.main:app"
	-pkill -f "streamlit run streamlit_app/Home.py"

# Everything, manual-mode: mlflow+api+streamlit as local processes + Airflow in
# Docker (Airflow has no manual mode — Postgres/scheduler/webserver need Docker).
manual-up-all: manual-up airflow-up

manual-down-all: manual-down airflow-down
