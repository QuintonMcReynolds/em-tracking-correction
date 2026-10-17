# Inference + validation image: ONNX Runtime and the package validator (no training stack).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt
COPY emtrack/__init__.py emtrack/package_runtime.py emtrack/
COPY package/ package/
RUN useradd -r -u 10001 app && chown -R app /app
USER app
CMD ["python", "-m", "emtrack.package_runtime", "package"]
