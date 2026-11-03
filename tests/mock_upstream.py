"""Mock: API OpenAI (9100) + llama-server falso (9200), para testar o llmrouter."""
import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def sse(obj):
    return ("data: " + json.dumps(obj) + "\n\n").encode()


class Base(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _body(self):
        n = int(self.headers.get("content-length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _send(self, code, raw, ctype="application/json", gz=False):
        if gz:
            raw = gzip.compress(raw)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class Api(Base):
    """Upstream estilo DeepSeek/OpenAI. Gzip se o cliente aceitar."""

    def do_POST(self):
        body = self._body()
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                    "created": 1, "model": "mock-api"}
            self.wfile.write(sse({**base, "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}))
            for piece in ("resposta ", "da ", "api"):
                self.wfile.write(sse({**base, "choices": [
                    {"index": 0, "delta": {"content": piece}, "finish_reason": None}]}))
            self.wfile.write(sse({**base, "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"}]}))
            self.wfile.write(sse({**base, "choices": [], "usage": {
                "prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107}}))
            self.wfile.write(b"data: [DONE]\n\n")
            return
        data = {"id": "chatcmpl-mock", "object": "chat.completion", "created": 1,
                "model": "mock-api",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "resposta da api"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107}}
        gz = "gzip" in (self.headers.get("accept-encoding") or "")
        self._send(200, json.dumps(data).encode(), gz=gz)


class Llama(Base):
    """llama-server falso: /health e SSE com timings no ultimo chunk."""

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b'{"status":"ok"}')
        else:
            self._send(404, b"{}")

    def do_POST(self):
        body = self._body()
        texto = json.dumps(body.get("messages", []))
        if "BOOM" in texto:  # simula crash do modelo local
            self._send(500, b'{"error":"model crashed"}')
            return
        if not body.get("stream"):
            data = {"choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "resposta local"}}],
                    "usage": {"prompt_tokens": 112, "completion_tokens": 9},
                    "timings": {"prompt_n": 112, "predicted_n": 9,
                                "predicted_per_second": 178.6}}
            self._send(200, json.dumps(data).encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for piece in ("resposta ", "local"):
            self.wfile.write(sse({"choices": [
                {"index": 0, "delta": {"content": piece}, "finish_reason": None}]}))
        self.wfile.write(sse({"choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"}]}))
        self.wfile.write(sse({"choices": [], "usage": {
            "prompt_tokens": 112, "completion_tokens": 9,
            "timings": {"prompt_n": 112, "predicted_n": 9,
                        "predicted_per_second": 178.6}}}))
        self.wfile.write(b"data: [DONE]\n\n")


def serve(port, handler):
    ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()


import sys

modo = sys.argv[1] if len(sys.argv) > 1 else "both"
if modo == "api":
    serve(9100, Api)
elif modo == "llama":
    serve(9200, Llama)
else:
    threading.Thread(target=serve, args=(9100, Api), daemon=True).start()
    serve(9200, Llama)
