FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[mcp]"

EXPOSE 8080
CMD ["python", "mcp_server.py", "--http", "--port", "8080"]
