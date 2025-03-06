# syntax=docker/dockerfile:1.2
FROM python:latest
# put you docker configuration here
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

RUN mkdir -p models

ENV PORT=8080

EXPOSE 8080

CMD ["python", "api.py"]