FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest

# 1) System packages and Microsoft ODBC Driver 17 for SQL Server
USER root
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    apt-transport-https \
    ca-certificates \
    build-essential \
    unixodbc \
    unixodbc-dev \
    libpq-dev \
    libssl-dev \
    libkrb5-3 \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Add Microsoft repo and install ODBC Driver 17 (needed by pyodbc)
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list -o /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql17 && \
    rm -rf /var/lib/apt/lists/*

# 2) Python setup
RUN python -m pip install --upgrade pip setuptools wheel

# 3) Workdir and project files
WORKDIR /app
COPY . /app

# 4) Install Python dependencies
# Include azure-ai-ml explicitly because it is imported by register/deploy scripts
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir azure-ai-ml azureml-inference-server-http

# Optional: ensure matplotlib/seaborn work in headless environments
ENV MPLCONFIGDIR=/tmp/mpl
RUN mkdir -p /tmp/mpl

# 5) Azure ML Inference config
# This helps local runs; Azure ML managed endpoints also support entry_script in deployment YAML
ENV AZUREML_ENTRY_SCRIPT=endpoint/score.py
ENV AZUREML_MODEL_DIR=/var/azureml-app/azureml-models

# 6) Expose HTTP port for local testing (Azure ML handles this internally)
EXPOSE 5001

# 7) Default to starting Azure ML inference server
# For training/pipeline jobs, Azure ML will override the command based on your component YAML.
CMD ["python", "-m", "azureml_inference_server_http"]
