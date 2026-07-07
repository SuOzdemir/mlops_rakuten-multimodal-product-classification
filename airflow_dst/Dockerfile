FROM apache/airflow:2.10.0

# CPU-only torch — training runs on CPU inside this container
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
