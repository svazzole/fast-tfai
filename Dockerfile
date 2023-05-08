FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# FROM nvidia/cuda:11.5.2-runtime-ubuntu20.04 AS base
# ENV PYTHON_VERSION=3.9.10

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

RUN apt-get update && apt-get -y upgrade
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3.8-venv python3-pip \
        ffmpeg libsm6 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /data
RUN mkdir /app

FROM base AS builder

ENV POETRY_VERSION=1.4.1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl  make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev \
        wget ca-certificates llvm libncurses5-dev \
        xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev \
        liblzma-dev mecab-ipadic-utf8 \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python3 -
ENV PATH "/root/.local/bin:$PATH"
RUN poetry completions bash >> ~/.bash_completion
RUN poetry config installer.max-workers 10
RUN poetry config virtualenvs.in-project true

WORKDIR /app
COPY pyproject.toml /app/
COPY poetry.lock /app/
RUN poetry install --only main --no-root --no-cache
COPY ./fast_tfai /app/fast_tfai
RUN poetry build
RUN poetry run pip install dist/*.whl
RUN rm -rf dist/

FROM base AS final

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY ./fast_tfai /app/fast_tfai
ENV PATH "./.venv/bin:$PATH"

CMD [ "train", "--conf", "conf/params.yaml"]
