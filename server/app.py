# server/app.py
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/reset":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress verbose logs

def main():
    port = 7860
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Serving on port {port}", flush=True)
    server.serve_forever()

if __name__ == "__main__":
    main()