import http.server
import socketserver
import requests
from urllib.parse import urlparse, parse_qs

PORT = 8001
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


class ForwardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(405)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Method Not Allowed')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        headers = {'Content-Type': 'application/json'}
        response = requests.post(OLLAMA_URL, headers=headers, data=post_data)

        self.send_response(response.status_code)
        for key, value in response.headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(response.content)


with socketserver.TCPServer(("", PORT), ForwardHandler) as httpd:
    print(f"Serving HTTP on port {PORT} (http://localhost:{PORT}/) ...")
    httpd.serve_forever()
