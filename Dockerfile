FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python pipeline/seed_db.py && python pipeline/etl.py && python pipeline/production_ml.py
CMD ["python", "app/server.py"]
