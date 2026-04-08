FROM python:3.11-slim

WORKDIR /app

COPY . .

# Install uv
RUN pip install uv

# Install dependencies using uv.lock
RUN uv sync

CMD ["sh", "-c", "python inference.py && python server/app.py"]
