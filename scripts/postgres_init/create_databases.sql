\getenv mlflow_password MLFLOW_DB_PASSWORD
\getenv airflow_password AIRFLOW_DB_PASSWORD
\getenv api_password API_DB_PASSWORD

CREATE USER mlflow WITH PASSWORD :'mlflow_password';
CREATE DATABASE mlflow OWNER mlflow;

CREATE USER airflow WITH PASSWORD :'airflow_password';
CREATE DATABASE airflow OWNER airflow;

CREATE USER api WITH PASSWORD :'api_password';
CREATE DATABASE api OWNER api;
