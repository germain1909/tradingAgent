# Use official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for ta-lib
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install prebuilt ta-lib binary
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && make && make install && \
    cd .. && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy your requirements.txt and install Python packages
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy your app and scripts
COPY . .

# Expose the port your Flask app will run on
EXPOSE 5000

# Default command to run Flask app
CMD ["python", "app.py"]
