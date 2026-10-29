# Dashboard image. Needs the NVIDIA container runtime (`docker run --gpus all`) for GPU use;
# without it the same image runs the pipeline on CPU. NOT BUILT OR RUN in the authoring
# environment (see CLAIMS.md).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY app ./app
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
