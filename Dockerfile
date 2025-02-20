#docker build -t goesgcp .
#docker save -o goesgcp.tar goesgcp
#docker load -i goesgcp.tar
#docker run -it goesgcp /bin/bash

# Usa a versão mínima do Debian como base
FROM debian:bookworm-slim

# Define o ambiente como não interativo para evitar prompts
ENV DEBIAN_FRONTEND=noninteractive

# Atualiza os pacotes e instala dependências essenciais
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    cdo \
    nco \
    build-essential \
    libssl-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    libreadline-dev \
    libncursesw5-dev \
    zlib1g-dev \
    xz-utils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Baixa e compila o Python 3.12
RUN wget https://www.python.org/ftp/python/3.12.1/Python-3.12.1.tgz && \
    tar -xf Python-3.12.1.tgz && \
    cd Python-3.12.1 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && rm -rf Python-3.12.1 Python-3.12.1.tgz

# Instala o pip para o Python 3.12
RUN python3.12 -m ensurepip && \
    python3.12 -m pip install --no-cache-dir --upgrade pip

# Instala o pacote goesgcp (branch labren) via pip
RUN python3.12 -m pip install --no-cache-dir git+https://github.com/helvecioneto/goesgcp.git@labren

# Define o Python 3.12 como o padrão
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1
