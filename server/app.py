from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from urllib.parse import urlparse, parse_qs

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path.endswith("/reset"):   # ✅ FIXED
            self._send_ok()
        elif path == "/" and query.get("logs") == ["container"]:
            self._send_logs_ok()
        elif path == "/":
            self._send_html()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.endswith("/reset"):   # ✅ FIXED
            self._send_ok()
        else:
            self.send_response(404)
            self.end_headers()

    def _send_ok(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())

    def _send_logs_ok(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Logs endpoint OK")

    def _send_html(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h1>GuardianNet is running</h1><p>Ready for evaluation.</p>")

def main():
    port = 7860
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Serving on port {port}", flush=True)
    server.serve_forever()

if __name__ == "__main__":
    main()
