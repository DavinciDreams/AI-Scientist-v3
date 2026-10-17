FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend
COPY apps/run_inspector/frontend/package*.json ./
RUN npm ci
COPY apps/run_inspector/frontend/ ./
RUN npm run typecheck && npm test && npm run build

FROM ghcr.io/astral-sh/uv:0.8.11 AS uv
FROM docker:28-cli AS docker-cli
FROM python:3.12-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=frontend /usr/local/bin/node /usr/local/bin/node
COPY --from=frontend /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
RUN apt-get update && apt-get install -y --no-install-recommends bash git procps ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/local/bin/python3 /usr/bin/python3
WORKDIR /opt/source
COPY . .
RUN python -c "from pathlib import Path; [p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n')) for p in Path('.').rglob('*') if p.is_file() and (p.suffix in ('.py', '.sh') or p.name == 'agsci-zai')]"
RUN uv sync --frozen --no-dev
COPY --from=frontend /frontend/dist ./apps/run_inspector/frontend/dist
COPY deploy/coolify/agsci-zai /usr/local/bin/agsci-zai
RUN chmod 755 /usr/local/bin/agsci-zai
ARG SOURCE_REVISION=local
ENV SOURCE_REVISION=$SOURCE_REVISION \
    PATH=/opt/source/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/data/sakanascientist/home \
    AGENTIC_TEMP_ROOT=/data/sakanascientist/tmp \
    TMPDIR=/data/sakanascientist/tmp
EXPOSE 8765
ENTRYPOINT ["python", "/opt/source/deploy/coolify/entrypoint.py"]
