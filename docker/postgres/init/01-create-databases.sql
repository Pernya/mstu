CREATE USER airflow WITH PASSWORD 'airflow';
CREATE DATABASE airflow OWNER airflow;
CREATE USER retail_user WITH PASSWORD 'retail_password';
CREATE DATABASE retail_dwh OWNER retail_user;
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
GRANT ALL PRIVILEGES ON DATABASE retail_dwh TO retail_user;
