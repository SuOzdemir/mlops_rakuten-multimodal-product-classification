.PHONY: install up serve down logs health airflow-up airflow-down airflow-logs test

# Local dev/training environment (pyproject.toml): notebooks, training scripts
install:
	uv sync

# Serving stack: mlflow -> api -> streamlit (order handled by depends_on/healthcheck)
up:
	docker compose up --build -d

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
