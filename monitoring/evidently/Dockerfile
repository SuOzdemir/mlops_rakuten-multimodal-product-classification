FROM python:3.12-slim

WORKDIR /app

COPY monitoring/evidently/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY monitoring/evidently/app.py /app/app.py
COPY src/api/reference_class_distribution.json /app/reference_class_distribution.json

EXPOSE 8002
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
