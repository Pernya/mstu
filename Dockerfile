FROM apache/airflow:2.10.5-python3.11

COPY requirements.txt /tmp/retail_requirements.txt

RUN pip install --no-cache-dir -r /tmp/retail_requirements.txt
