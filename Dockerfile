FROM node:22-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHON_BIN="python"
ENV NODE_ENV="production"
ENV PORT=7860

COPY package*.json ./
COPY server/package*.json ./server/
COPY client/package*.json ./client/

RUN npm ci --prefix server \
    && npm ci --prefix client

COPY src ./src
COPY scripts ./scripts
COPY models ./models
COPY server ./server
COPY client ./client

RUN npm --prefix client run build

WORKDIR /app/server

EXPOSE 7860

CMD ["node", "src/index.js"]
