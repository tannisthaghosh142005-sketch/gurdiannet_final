FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Run inference once, then start the web server using the entry point
CMD python inference.py && guardiannet-server
