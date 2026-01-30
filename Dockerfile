# Use docker.1ms.run as registry mirror for base image
FROM docker.1ms.run/python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Set pip source to USTC
    PIP_INDEX_URL=https://pypi.mirrors.ustc.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.mirrors.ustc.edu.cn

# Set work directory to /app/agent_service so imports work correctly
WORKDIR /app/agent_service

# Replace APT sources with USTC mirror for Debian Bookworm (Python 3.13 slim is based on bookworm)
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources

# Install system dependencies
# build-essential for compiling some python packages if wheels are missing
# curl/wget for healthchecks if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Move back to /app to run the command, so that 'agent_service' is in the python path
WORKDIR /app

# Expose port
EXPOSE 8002

# Command to run the application using module syntax or setting PYTHONPATH
# We set PYTHONPATH to include /app
ENV PYTHONPATH=/app

CMD ["python", "agent_service/server.py"]
