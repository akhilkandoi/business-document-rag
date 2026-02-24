# Use python 3.11 slim image as base image
FROM python:3.11-slim

#set working directory
WORKDIR /app

#install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

#copy requirements first for better caching
COPY requirements.txt .

#install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#copy application code
COPY . .

#expose ports
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# default command
CMD ["python","api.py"]
