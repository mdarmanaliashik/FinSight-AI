"""
FinSight AI - Production Web Server for Render & Cloud Deployment
=================================================================
Serves the high-aesthetic Executive Decision Intelligence Dashboard (dashboard/index.html)
with correct MIME types, caching headers, and health check support.
"""

import http.server
import socketserver
import os
import sys

PORT = int(os.environ.get("PORT", 8501))
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")

class FinSightHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Enable CORS and disable aggressive caching for real-time updates
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()

    def do_GET(self):
        # Health check endpoint for Render / Docker health checks
        if self.path in ("/health", "/_stcore/health", "/healthz"):
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
            return
        return super().do_GET()

if __name__ == "__main__":
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print(f"[*] FinSight AI Server launching on port {PORT}...")
    print(f"[*] Serving web dashboard from: {DIRECTORY}")
    
    # Allow port reuse to prevent address already in use errors on restart
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), FinSightHTTPHandler) as httpd:
        print(f"[OK] Dashboard live at: http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.server_close()
