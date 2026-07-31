FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python seed_db.py && python etl.py && python production_ml.py
CMD ["python", "server.py"]
