.PHONY: install setup-data prepare-splits docker-up up serve up-all down down-all restart-all logs health airflow-up airflow-down airflow-logs evidently-up evidently-down evidently-logs test manual-up manual-down manual-up-all manual-down-all

# Ensure the Docker daemon is available. On macOS, start Docker Desktop when
# needed and wait up to two minutes for it to become ready.
docker-up:
	@if docker info >/dev/null 2>&1; then \
		echo "Docker is ready."; \
	elif [ "$$(uname -s)" = "Darwin" ]; then \
		echo "Docker Desktop is not running; starting it..."; \
		open -a Docker; \
		attempt=0; \
		until docker info >/dev/null 2>&1; do \
			attempt=$$((attempt + 1)); \
			if [ $$attempt -ge 60 ]; then \
				echo "ERROR: Docker Desktop did not become ready within 120 seconds."; \
				exit 1; \
			fi; \
			sleep 2; \
		done; \
		echo "Docker Desktop is ready."; \
	else \
		echo "ERROR: Docker daemon is not running. Start Docker, then retry."; \
		exit 1; \
	fi

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
# Refuses to start if manual-up's native processes already hold the same ports
# (5001/8000/8501) — both modes bound to the same port at once means whichever
# one didn't win the socket silently never receives traffic, e.g. mlflow's
# Host-header check rejecting requests that actually reached the *other* mlflow.
up: docker-up
	@if pgrep -f "mlflow server" >/dev/null || pgrep -f "uvicorn src.api.main:app" >/dev/null || pgrep -f "streamlit run streamlit_app/Home.py" >/dev/null; then \
		echo "ERROR: manual-up's native processes (mlflow/api/streamlit) are still running on ports 5001/8000/8501."; \
		echo "Run 'make manual-down' first, or use manual-up-all instead of the Docker stack."; \
		exit 1; \
	fi
	docker compose up --build -d --wait

# Everything: serving stack (mlflow+api+streamlit) + Airflow (own compose stack/network)
up-all: up airflow-up

down-all: down airflow-down

# Full stop then full start in one command -- for after a Docker Desktop
# restart, a host resource crunch, or just "did I break something, reset
# and check" without typing two commands.
restart-all: down-all up-all

# Just api (:8000) + streamlit (:8501), no mlflow — same ports every time
serve: docker-up
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
airflow-up: docker-up
	docker compose --profile promotion build promotion
	cd airflow_dst && docker compose --env-file ../.env up --build -d

airflow-down:
	cd airflow_dst && docker compose --env-file ../.env down

airflow-logs:
	cd airflow_dst && docker compose --env-file ../.env logs -f

# Optional Evidently sidecar. Rebuild the API first so prediction_events are
# persisted, then start the monitor without replacing the existing PSI path.
evidently-up: docker-up
	docker compose up --build -d --wait api
	docker compose --profile evidently up --build -d --wait evidently-monitor

evidently-down:
	docker compose --profile evidently stop evidently-monitor

evidently-logs:
	docker compose --profile evidently logs -f evidently-monitor

test:
	uv run pytest

# Same serving stack as `up`, but without Docker: mlflow + api + streamlit as
# background local processes (uv run), logs go to *.log (gitignored).
# Refuses to start if the Docker stack is already up on the same ports — see
# the matching guard on `up` for why running both at once is broken, not just
# wasteful.
manual-up: docker-up
	@if [ -n "$$(docker compose ps -q mlflow api streamlit 2>/dev/null)" ]; then \
		echo "ERROR: docker compose stack (mlflow/api/streamlit) is already running on ports 5001/8000/8501."; \
		echo "Run 'make down' first, or use the Docker stack instead of manual-up."; \
		exit 1; \
	fi
	# Same Postgres (published on localhost:5432 by the `postgres` service) and
	# the same ./mlruns/artifacts dir as the Docker mlflow -- one shared run
	# history regardless of which mode started it, instead of a second,
	# disconnected SQLite-backed mlflow.
	set -a; . ./.env; set +a; PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python uv run mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri postgresql://mlflow:$$MLFLOW_DB_PASSWORD@localhost:5432/mlflow --artifacts-destination ./mlruns/artifacts --serve-artifacts > mlflow.log 2>&1 &
	set -a; . ./.env; set +a; MLFLOW_TRACKING_URI=http://localhost:5001 AIRFLOW_API_URL=http://localhost:8080 DATABASE_URL=postgresql://api:$$API_DB_PASSWORD@localhost:5432/api uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
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
