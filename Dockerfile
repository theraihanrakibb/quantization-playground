FROM python:3.11-slim

# Run as a non-root user
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER appuser

# Smoke-test entrypoint: run the toy MLP comparison with FP8
ENTRYPOINT ["python", "-m", "quant_playground.cli"]
CMD ["demo", "--format", "fp8"]
