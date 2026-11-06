FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app/densea-rfdetr

# Pinned dependencies (see requirements-pinned.txt), then the package itself.
COPY requirements-pinned.txt pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-pinned.txt && \
    pip install --no-cache-dir --no-deps -e .

COPY densea/ densea/
COPY tests/ tests/

CMD ["sleep", "infinity"]
