FROM --platform=linux/arm64 python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["python", "-m", "adlc_review_agent.agentcore_entrypoint"]
