FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot/ ./bot/
COPY shared/ ./shared/
COPY ml/ ./ml/
COPY data/ ./data/

CMD ["python", "bot/main.py"]
