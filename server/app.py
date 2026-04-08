from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from urllib.parse import urlparse, parse_qs

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # Handle /reset endpoint (required for hackathon)
        if path == "/reset":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        
        # Handle /?logs=container (internal HF health check)
        elif path == "/" and query.get("logs") == ["container"]:
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Logs endpoint OK")
        
        # Handle root path (optional – shows a simple page)
        elif path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>GuardianNet is running</h1><p>Ready for evaluation.</p>")
        
        # All other paths return 404
        else:
            self.send_response(404)
            self.end_headers()

def main():
    port = 7860
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Serving on port {port}", flush=True)
    server.serve_forever()

if __name__ == "__main__":
    main()
