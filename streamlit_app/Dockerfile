FROM python:3.12-slim

WORKDIR /project

COPY streamlit_app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY streamlit_app/Home.py       /project/streamlit_app/Home.py
COPY streamlit_app/__init__.py   /project/streamlit_app/__init__.py

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app/Home.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
