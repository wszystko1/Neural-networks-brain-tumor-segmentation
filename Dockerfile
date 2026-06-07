FROM runpod/base:1.0.2-cuda1290-ubuntu2204

ENV PYTHONUNBUFFERED=1
ENV UV_NO_DEV=1

WORKDIR /app

RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        wget curl git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --locked

COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]
