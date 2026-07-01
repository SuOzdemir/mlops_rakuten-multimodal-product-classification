FROM python:3.12-slim

WORKDIR /project

# CPU-only torch — keeps image small, GPU not needed for inference
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY src/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Package structure the API needs to import from
COPY src/__init__.py              /project/src/__init__.py
COPY src/api/                     /project/src/api/
COPY streamlit_app/__init__.py    /project/streamlit_app/__init__.py
COPY streamlit_app/services/      /project/streamlit_app/services/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
