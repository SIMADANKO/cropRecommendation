import http.server
import socketserver

PORT = 8080
DIRECTORY = "."

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # 设置CORS头，允许跨域访问
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"前端服务运行在 http://localhost:{PORT}")
        print("可以通过浏览器访问 http://localhost:8080/index.html")
        httpd.serve_forever()