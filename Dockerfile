
FROM python:3.10-slim

WORKDIR /app

# System deps (build tools + libgl for some packages)
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
