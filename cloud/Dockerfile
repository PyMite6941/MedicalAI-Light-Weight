FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY cloud/requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

COPY cloud/train_entry.py cloud/data_pipeline.py ./
COPY training_fusion.py .
COPY expand_dataset.py .

ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

ENTRYPOINT ["python", "train_entry.py"]