#!/usr/bin/env python3
"""
quality_bench.py (v3) - mede qualidade (acerto) e tempo ate a resposta final de
modelos GGUF locais, um por vez, via llama-server, e explica POR QUE cada
tarefa falhou ou demorou.

Para cada modelo: resolve/baixa o arquivo -> garante a GPU limpa -> sobe o
llama-server -> roda as tarefas -> derruba o servidor e VERIFICA a descarga
(VRAM, porta e processos) -> apaga o arquivo baixado (a menos que --keep).

Layout dos samples (um por teste, prompt + resposta completos):
    ~/quality-details/<AAAA-MM-DD>/<modelo>-<commit>/<tarefa>-run<N>.json
    ~/quality-details/<AAAA-MM-DD>/<modelo>-<commit>/manifest.json
    ~/quality-details/<AAAA-MM-DD>/<modelo>-<commit>/llama-server.log

O <commit> e uma hash aleatoria de 40 hex (estilo git), unica por execucao
(use --commit para reaproveitar uma). A pasta do dia e reaproveitada se ja
existir. Cada JSON guarda a requisicao inteira (messages + tools), o prompt,
a resposta final completa, o raciocinio, tool_calls e a resposta bruta da API.

Uso:
    python quality_bench.py                          # todos os modelos
    python quality_bench.py --only gpt-oss --tasks k8s   # so uma tarefa
    python quality_bench.py --runs 5 --max-tokens 12000 --fail-fast
    python quality_bench.py --backend cpu            # forca CPU+RAM
    python quality_bench.py --ngl all --backend gpu  # forca 100% GPU

Requisitos: python 3 (stdlib). PyYAML e opcional (so a tarefa de K8s usa).
ATENCAO: o script executa codigo gerado pelo modelo em subprocessos com
timeout, mas SEM sandbox.
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import secrets
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

# ----------------------------- configuracao --------------------------------
LLAMA_SERVER = os.path.expanduser("~/ai/llama.cpp/build/bin/llama-server")
DEVICE = "Vulkan0"                          # RX 9070 XT; Vulkan1 e a iGPU
PORT = 8099
CTX = 16384
DL_DIR = os.path.expanduser("~/models/tmp")
HUB = os.path.expanduser("~/.cache/huggingface/hub")
OUT_MD = os.path.expanduser("~/quality-catalog.md")
DETAILS_ROOT = os.path.expanduser("~/quality-details")
ROCM_SMI = shutil.which("rocm-smi")
VRAM_CARD = 0                               # card0 = dGPU no rocm-smi
VRAM_TOL = 1024 ** 3                        # folga de 1 GiB p/ "VRAM liberada"
UNLOAD_TIMEOUT = 90

MODELS = [
    dict(name="gpt-oss-20b-Q4XL",
         glob=f"{HUB}/models--unsloth--gpt-oss-20b-GGUF/snapshots/*/gpt-oss-20b-UD-Q4_K_XL.gguf"),
    dict(name="gemma-4-12b-Q4XL", repo="unsloth/gemma-4-12b-it-GGUF",
         file="gemma-4-12b-it-UD-Q4_K_XL.gguf"),
    dict(name="qwen3.5-9b-Q4XL", repo="unsloth/Qwen3.5-9B-GGUF",
         file="Qwen3.5-9B-UD-Q4_K_XL.gguf"),
]


# ------------------------------- infraestrutura -----------------------------
class TaskAbort(Exception):
    """Erro de infraestrutura (timeout, HTTP, conexao): aborta a tarefa."""


class Resp:
    def __init__(self, data, dt, prompt="", request=None):
        msg = data["choices"][0]["message"]
        self.content = msg.get("content") or ""
        self.reasoning = msg.get("reasoning_content") or ""
        self.tool_calls = msg.get("tool_calls") or []
        self.finish = data["choices"][0].get("finish_reason")
        u = data.get("usage") or {}
        t = data.get("timings") or {}
        self.toks = u.get("completion_tokens") or t.get("predicted_n") or 0
        self.ptoks = u.get("prompt_tokens") or t.get("prompt_n") or 0
        self.gen_tps = t.get("predicted_per_second")
        self.pp_tps = t.get("prompt_per_second")
        self.dt = dt
        self.prompt = prompt
        self.request = request
        self.raw = data

    def diag(self):
        tps = f"{self.gen_tps:.0f} tok/s" if self.gen_tps else "tok/s n/d"
        s = (f"{self.toks} tok gerados ({tps}), prompt {self.ptoks} tok, "
             f"finish={self.finish}, raciocinio={len(self.reasoning)} chars, "
             f"resposta={len(self.content)} chars")
        if self.finish == "length":
            s += " [ESTOUROU max_tokens: provavel loop ou raciocinio longo]"
        elif not self.content.strip() and not self.tool_calls:
            s += " [resposta vazia]"
        return s


def chat(messages, tools, temp, max_tokens, timeout, stream=False,
         live_reason=None, live_content=None):
    body = {"model": "x", "messages": messages, "temperature": temp,
            "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = (stream_response(r, live_reason, live_content) if stream
                    else json.load(r))
    except urllib.error.HTTPError as e:
        raise TaskAbort(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
    except (socket.timeout, TimeoutError):
        raise TaskAbort(f"timeout: sem resposta apos {timeout}s")
    except urllib.error.URLError as e:
        if isinstance(e.reason, (socket.timeout, TimeoutError)):
            raise TaskAbort(f"timeout: sem resposta apos {timeout}s")
        raise TaskAbort(f"erro de conexao: {e.reason}")
    return Resp(data, time.time() - t0, request=body)


class Live:
    """Escreve o texto conforme chega, uma linha por vez, com prefixo."""

    def __init__(self, prefix, dim=False):
        self.prefix = prefix
        self.color = "\033[2m" if (dim and sys.stdout.isatty()
                                   and not os.environ.get("NO_COLOR")) else ""
        self.reset = "\033[0m" if self.color else ""
        self.buf = ""
        self.chars = 0

    def feed(self, text):
        self.chars += len(text)
        self.buf += text.replace("\r\n", "\n").replace("\r", "\n")
        *lines, self.buf = self.buf.split("\n")
        for line in lines:
            self._emit(line)

    def _emit(self, line):
        sys.stdout.write(f"{self.prefix}{self.color}{line}{self.reset}\n")
        sys.stdout.flush()

    def close(self):
        if self.buf:
            self._emit(self.buf)
            self.buf = ""
        sys.stdout.flush()


def stream_response(resp, live_reason=None, live_content=None):
    """Le o SSE do llama-server e remonta a resposta completa (sem perder nada)."""
    content, reasoning = [], []
    calls, finish = {}, None
    usage, timings, chunks = {}, {}, 0
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            ch = json.loads(payload)
        except ValueError:
            continue
        chunks += 1
        if ch.get("usage"):
            usage = ch["usage"]
        if ch.get("timings"):
            timings = ch["timings"]
        choice = (ch.get("choices") or [{}])[0] or {}
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
        d = choice.get("delta") or {}
        piece = d.get("reasoning_content") or d.get("reasoning") or ""
        if piece:
            reasoning.append(piece)
            if live_reason:
                live_reason.feed(piece)
        piece = d.get("content") or ""
        if piece:
            content.append(piece)
            if live_content:
                live_content.feed(piece)
        for tc in d.get("tool_calls") or []:
            i = tc.get("index", 0)
            slot = calls.setdefault(i, {"id": "", "type": "function",
                                        "function": {"name": "", "arguments": ""}})
            if tc.get("id"):
                slot["id"] = tc["id"]
            f = tc.get("function") or {}
            if f.get("name"):
                slot["function"]["name"] = f["name"]
            if f.get("arguments"):
                slot["function"]["arguments"] += f["arguments"]
    msg = {"role": "assistant", "content": "".join(content),
           "reasoning_content": "".join(reasoning)}
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    if live_reason:
        live_reason.close()
    if live_content:
        live_content.close()
    return {"choices": [{"message": msg, "finish_reason": finish}],
            "usage": usage, "timings": timings,
            "streamed": True, "chunks": chunks}


def run_py(code, timeout=15, cwd=None):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=cwd) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout, cwd=cwd)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    finally:
        os.unlink(path)


def last_line(s):
    lines = [l for l in (s or "").strip().splitlines() if l.strip()]
    return lines[-1][:200] if lines else "(sem saida)"


def fenced(text):
    return re.findall(r"```[\w+-]*\n(.*?)```", text or "", re.S)


def first_block(text):
    """Retorna (codigo, tinha_bloco)."""
    b = fenced(text)
    return (b[0], True) if b else (text or "", False)


def nofence(had):
    return "" if had else "sem bloco ``` na resposta; "


# --------------------------------- tarefas ----------------------------------
# Cada tarefa devolve (ok, motivo, resp). ok=None significa "tarefa pulada".
BUGGY_CHUNK = '''def chunk(items, size):
    """Divide a lista em pedacos de tamanho `size`; o ultimo pode ser menor."""
    out = []
    for i in range(0, len(items) - size, size):
        out.append(items[i:i + size])
    return out
'''


def t_bugfix(i, ask):
    r = ask("Esta funcao tem um bug (chunk([1,2,3,4,5], 2) deveria dar "
            "[[1,2],[3,4],[5]]). Responda com a funcao corrigida completa em um "
            "unico bloco ```python, sem mudar a assinatura.\n\n```python\n"
            + BUGGY_CHUNK + "```")
    code, had = first_block(r.content)
    tests = ("\nassert chunk([1,2,3,4,5],2)==[[1,2],[3,4],[5]]\n"
             "assert chunk([],3)==[]\nassert chunk([1,2,3,4],2)==[[1,2],[3,4]]\n"
             "assert chunk([1],5)==[[1]]\n")
    rc, _, err = run_py(code + tests)
    if rc == 0:
        return True, "", r
    why = "timeout executando o codigo (loop infinito?)" if rc == 124 \
        else "testes falharam: " + last_line(err)
    return False, nofence(had) + why, r


def t_surgical(i, ask):
    r = ask("Complete a funcao abaixo sem mudar o nome nem a assinatura. "
            "Responda so com a funcao completa em um bloco ```python.\n\n"
            "```python\ndef parse_duration(s):\n"
            '    """Converte \'1h30m\', \'45s\', \'2h\', \'1h2m3s\' em segundos (int).\n'
            '    Levanta ValueError se o texto for invalido."""\n'
            "    ...\n```")
    code, had = first_block(r.content)
    tests = '''
assert parse_duration("1h30m")==5400
assert parse_duration("45s")==45
assert parse_duration("2h")==7200
assert parse_duration("1h2m3s")==3723
for bad in ["abc", "", "10x", "h"]:
    try:
        parse_duration(bad)
    except ValueError:
        pass
    else:
        raise SystemExit("nao levantou ValueError para %r" % bad)
'''
    rc, _, err = run_py(code + tests)
    if rc == 0:
        return True, "", r
    why = "timeout executando o codigo (loop infinito?)" if rc == 124 \
        else "testes falharam: " + last_line(err)
    return False, nofence(had) + why, r


BUGGY_WS = '''def window_sums(xs, k):
    """Soma de cada janela deslizante de tamanho k."""
    return [sum(xs[i:i + k]) for i in range(len(xs) - k)]
'''
GOOD_WS = BUGGY_WS.replace("len(xs) - k)", "len(xs) - k + 1)")


def t_write_test(i, ask):
    r = ask("A funcao abaixo tem um bug. NAO a corrija. Escreva UMA funcao "
            "`test_window_sums()` com asserts que FALHE (AssertionError) com este "
            "codigo e PASSE com uma implementacao correta. Responda so com a "
            "funcao de teste em um bloco ```python.\n\n```python\n"
            + BUGGY_WS + "```")
    test, had = first_block(r.content)
    rc, out, err = run_py(
        BUGGY_WS + "\n" + test + "\ntry:\n    test_window_sums()\n"
        "except AssertionError:\n    print('FALHOU_COMO_ESPERADO')\n")
    if rc != 0:
        return False, nofence(had) + "teste quebrou no codigo com bug (erro que nao e " \
            "AssertionError): " + last_line(err), r
    if "FALHOU_COMO_ESPERADO" not in out:
        return False, nofence(had) + "teste PASSOU no codigo com bug (nao detecta o bug)", r
    rc2, _, err2 = run_py(GOOD_WS + "\n" + test + "\ntest_window_sums()\n")
    if rc2 != 0:
        return False, "teste FALHOU na implementacao correta: " + last_line(err2), r
    return True, "", r


A_SRC = '''def titulo_a(txt):
    s = txt.strip().lower()
    s = "".join(c if c.isalnum() else "-" for c in s)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")
'''
B_SRC = '''def titulo_b(txt, prefixo="post"):
    s = txt.strip().lower()
    s = "".join(c if c.isalnum() else "-" for c in s)
    while "--" in s:
        s = s.replace("--", "-")
    return prefixo + "/" + s.strip("-")
'''
REFACTOR_INPUTS = ["  Ola, Mundo!! ", "Ja--Foi  assim", "---x---", "", "A_B c"]


def t_refactor(i, ask):
    r = ask("Refatore: mova a logica duplicada para `common.py` com a funcao "
            "`slugify(txt)`, e reescreva a.py e b.py para usa-la, mantendo o "
            "comportamento publico identico. Responda com 3 blocos ```python, "
            "cada um comecando com a linha `# file: nome.py` (common.py, a.py, "
            "b.py), com a linha `# file:` DENTRO do bloco.\n\nOriginais:\n\n"
            "```python\n# file: a.py\n" + A_SRC + "```\n\n"
            "```python\n# file: b.py\n" + B_SRC + "```")
    files = {}
    for blk in fenced(r.content):
        m = re.match(r"\s*#\s*file:\s*(\S+)", blk)
        if m:
            files[m.group(1)] = blk
    missing = {"common.py", "a.py", "b.py"} - set(files)
    if missing:
        return False, f"arquivos ausentes na resposta: {sorted(missing)} " \
            f"(achou {sorted(files)})", r
    if ".replace(" in files["a.py"] or ".replace(" in files["b.py"]:
        return False, "a.py/b.py ainda contem a logica duplicada (.replace)", r
    ns = {}
    exec(A_SRC + "\n" + B_SRC, ns)
    expected = [[ns["titulo_a"](s), ns["titulo_b"](s), ns["titulo_b"](s, "x")]
                for s in REFACTOR_INPUTS]
    with tempfile.TemporaryDirectory() as d:
        for name, content in files.items():
            with open(os.path.join(d, os.path.basename(name)), "w") as f:
                f.write(content)
        probe = ("import json\nfrom a import titulo_a\nfrom b import titulo_b\n"
                 f"ins = {REFACTOR_INPUTS!r}\n"
                 "print(json.dumps([[titulo_a(s), titulo_b(s), titulo_b(s, 'x')]"
                 " for s in ins]))\n")
        rc, out, err = run_py(probe, cwd=d)
    if rc != 0:
        return False, "erro ao importar/executar: " + last_line(err), r
    try:
        got = json.loads(out)
    except ValueError:
        return False, "saida inesperada: " + out[:100], r
    if got != expected:
        for ins, g, e in zip(REFACTOR_INPUTS, got, expected):
            if g != e:
                return False, f"comportamento mudou para {ins!r}: esperado {e}, veio {g}", r
    return True, "", r


def t_longctx(i, ask):
    rng = random.Random(42)
    vals, lines = {}, []
    for n in range(300):
        v = rng.randint(10000, 99999)
        vals[n] = v
        lines.append(f"def calc_tarifa_{n:03d}(x):\n"
                     f"    return x * {rng.randint(2, 9)} + {v}\n")
    target = [41, 187, 262, 99, 230][i % 5]
    r = ask("Codigo:\n\n" + "\n".join(lines) +
            f"\n\nQual e a constante somada em calc_tarifa_{target:03d}? "
            "Responda so o numero.", max_tokens=4000)
    if str(vals[target]) in r.content:
        return True, "", r
    return False, f"esperado {vals[target]}; resposta: {r.content[:80]!r}", r


def dig(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        elif isinstance(d, list) and isinstance(k, int) and len(d) > k:
            d = d[k]
        else:
            return None
    return d


def t_k8s(i, ask):
    try:
        import yaml
    except ImportError:
        return None, "PyYAML ausente (pacote python-yaml no Arch)", None
    r = ask("Escreva um Deployment Kubernetes (apps/v1) chamado `web`: 3 replicas, "
            "label app=web, imagem nginx:1.27, container na porta 8080, requests "
            "cpu 100m e memoria 128Mi, limits cpu 500m e memoria 256Mi, "
            "readinessProbe HTTP GET /healthz na porta 8080. Responda so com o "
            "YAML em um bloco ```yaml.")
    text, had = first_block(r.content)
    try:
        docs = [d for d in yaml.safe_load_all(text) if isinstance(d, dict)]
    except Exception as e:
        return False, nofence(had) + "YAML invalido: " + str(e).splitlines()[0][:150], r
    deps = [d for d in docs if d.get("kind") == "Deployment"]
    if not deps:
        kinds = [d.get("kind") for d in docs]
        return False, nofence(had) + f"nenhum documento Deployment (kinds: {kinds})", r
    d = deps[0]
    c = dig(d, "spec", "template", "spec", "containers", 0) or {}
    labels = dict(dig(d, "metadata", "labels") or {})
    labels.update(dig(d, "spec", "template", "metadata", "labels") or {})
    probs = []

    def chk(cond, msg):
        if not cond:
            probs.append(msg)

    chk(dig(d, "metadata", "name") == "web", f"name={dig(d, 'metadata', 'name')!r}")
    chk(dig(d, "spec", "replicas") == 3, f"replicas={dig(d, 'spec', 'replicas')!r}")
    chk(labels.get("app") == "web", f"label app ausente (labels={labels})")
    chk(c.get("image") == "nginx:1.27", f"image={c.get('image')!r}")
    ports = [p.get("containerPort") for p in (c.get("ports") or []) if isinstance(p, dict)]
    chk(8080 in ports, f"containerPort 8080 ausente (ports={ports})")
    chk(dig(c, "resources", "requests", "cpu") == "100m", "requests.cpu != 100m")
    chk(dig(c, "resources", "requests", "memory") == "128Mi", "requests.memory != 128Mi")
    chk(dig(c, "resources", "limits", "cpu") == "500m", "limits.cpu != 500m")
    chk(dig(c, "resources", "limits", "memory") == "256Mi", "limits.memory != 256Mi")
    chk(dig(c, "readinessProbe", "httpGet", "path") == "/healthz", "readinessProbe.path != /healthz")
    chk(str(dig(c, "readinessProbe", "httpGet", "port")) == "8080", "readinessProbe.port != 8080")
    if probs:
        return False, nofence(had) + "; ".join(probs), r
    return True, "", r


CITIES = ["Lisboa", "Tokyo", "Fortaleza", "Berlin", "Oslo",
          "Nairobi", "Lima", "Sydney", "Toronto", "Cairo"]
MONTHS = ["janeiro", "fevereiro", "marco", "abril", "maio",
          "junho", "julho", "agosto", "setembro", "outubro"]
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Clima atual de uma cidade",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "run_sql", "description": "Executa uma query SQL somente leitura",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]


def t_tools(ask):
    """20 chamadas. Retorna (acertos, total, respostas, motivos, casos)."""
    cases = [(f"Qual o clima agora em {c}?", "get_weather", "city", c) for c in CITIES]
    cases += [(f"Conte quantos pedidos da tabela orders foram feitos em {m}.",
               "run_sql", "query", "orders") for m in MONTHS]
    ok, resps, reasons, casos = 0, [], [], []
    for idx, (prompt, fn, key, needle) in enumerate(cases, 1):
        try:
            r = ask(prompt, tools=TOOLS, max_tokens=4000,
                    label=f"tool calling {idx}/{len(cases)}")
        except TaskAbort as e:
            reasons.append(str(e))
            casos.append({"i": idx, "prompt": prompt, "ok": False,
                          "why": str(e), "resp": None})
            break
        resps.append(r)
        why = None
        call = r.tool_calls[0] if r.tool_calls else None
        if not call:
            why = "nao chamou tool (respondeu em texto)" if r.content.strip() else "sem tool_call e sem texto"
        elif call["function"]["name"] != fn:
            why = f"tool errada: {call['function']['name']} (esperado {fn})"
        else:
            try:
                args = json.loads(call["function"]["arguments"])
                if needle.lower() not in str(args.get(key, "")).lower():
                    why = f"argumento {key!r} sem {needle!r}: {args}"
            except ValueError:
                why = "arguments nao e JSON valido: " + str(call["function"]["arguments"])[:100]
        if why:
            reasons.append(why)
            casos.append({"i": idx, "prompt": prompt, "ok": False,
                          "why": why, "resp": r, "diag": r.diag()})
        else:
            ok += 1
            casos.append({"i": idx, "prompt": prompt, "ok": True,
                          "why": "", "resp": r})
    return ok, len(cases), resps, reasons, casos


TASKS = [("bug com teste", t_bugfix), ("edicao cirurgica", t_surgical),
         ("escrever teste", t_write_test), ("refactor multi-arquivo", t_refactor),
         ("contexto longo (~9k tok)", t_longctx), ("k8s deployment", t_k8s)]


# ------------------------------- GPU / servidor -----------------------------
def gpu_used(card=VRAM_CARD):
    """VRAM usada em bytes (None se rocm-smi indisponivel)."""
    if not ROCM_SMI:
        return None
    try:
        out = subprocess.run([ROCM_SMI, "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(json.loads(out)[f"card{card}"]["VRAM Total Used Memory (B)"])
    except Exception:
        return None


def gpu_total(card=VRAM_CARD):
    if not ROCM_SMI:
        return None
    try:
        out = subprocess.run([ROCM_SMI, "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(json.loads(out)[f"card{card}"]["VRAM Total Memory (B)"])
    except Exception:
        return None


def gib(n):
    return "n/d" if n is None else f"{n / 1024 ** 3:.2f} GiB"


class Gpu:
    """Baseline/pico de VRAM enquanto o modelo esta carregado."""

    def __init__(self, interval=3.0):
        self.interval = interval
        self.baseline = gpu_used()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._th = None

    def start(self):
        self.baseline = gpu_used()
        self.peak = self.baseline
        self._stop.clear()

        def loop():
            while not self._stop.wait(self.interval):
                u = gpu_used()
                if u is not None and (self.peak is None or u > self.peak):
                    self.peak = u

        self._th = threading.Thread(target=loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=30)
        return self.baseline, self.peak, gpu_used()


def port_open(port=PORT):
    s = socket.socket()
    s.settimeout(0.5)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def pids_on_port(port=PORT):
    pids = []
    try:
        out = subprocess.run(["ss", "-lptnH", f"sport = :{port}"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return pids
    for m in re.finditer(r"pid=(\d+)", out):
        p = int(m.group(1))
        if p not in pids:
            pids.append(p)
    return pids


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().decode(errors="replace").replace("\0", " ").strip()
    except Exception:
        return ""


def kill_llama_remnants():
    """Mata llama-server que ainda estejam presos nesta porta (so os da bench)."""
    killed = []
    for p in pids_on_port():
        c = cmdline(p)
        if "llama-server" in c:
            try:
                os.kill(p, signal.SIGKILL)
                killed.append(p)
            except ProcessLookupError:
                pass
    return killed


def stop_and_verify(proc, sess, gpu):
    """Derruba o servidor e confirma que VRAM/porta/processos voltaram ao normal."""
    info = {"problemas": [], "porta_livre": False}

    if proc is not None:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                proc.terminate()
            try:
                proc.wait(timeout=30)
                info["encerrado"] = "SIGTERM"
            except subprocess.TimeoutExpired:
                info["problemas"].append("SIGTERM nao encerrou em 30s; enviando SIGKILL")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=15)
                    info["encerrado"] = "SIGKILL"
                except subprocess.TimeoutExpired:
                    info["problemas"].append("processo nao morreu nem com SIGKILL")
        else:
            info["encerrado"] = f"ja tinha saido (rc={proc.returncode})"

    t0 = time.time()
    while port_open() and time.time() - t0 < 20:
        time.sleep(0.25)
    info["porta_livre"] = not port_open()
    if not info["porta_livre"]:
        info["problemas"].append(f"porta {PORT} ainda aberta")

    # VRAM de volta ao baseline (com folga), senao cacar processo agarrado na GPU.
    base = gpu.baseline
    used = gpu_used()
    deadline = time.time() + UNLOAD_TIMEOUT
    while (base is not None and used is not None
           and used - base > VRAM_TOL and time.time() < deadline):
        time.sleep(2)
        used = gpu_used()
    if base is not None and used is not None and used - base > VRAM_TOL:
        killed = kill_llama_remnants()
        if killed:
            info["problemas"].append(
                f"VRAM presa ({gib(used)} vs baseline {gib(base)}); matei pids {killed}")
            time.sleep(3)
            used = gpu_used()
    info["vram_baseline"] = base
    info["vram_final"] = used
    if base is not None and used is not None and used - base > VRAM_TOL:
        info["problemas"].append(
            f"VRAM NAO voltou ao baseline: {gib(used)} (baseline {gib(base)})")
        info["pids_na_porta"] = [(p, cmdline(p)) for p in pids_on_port()]

    ok = not info["problemas"]
    if ok:
        print(f"   unload ok: {info.get('encerrado', 'sem processo')} | "
              f"porta {PORT} livre | VRAM {gib(used)} (baseline {gib(base)}, "
              f"pico {gib(gpu.peak)})")
    else:
        for p in info["problemas"]:
            print(f"   !! unload: {p}")
    sess.unload = info
    return ok


def ensure_gpu_clean(target, rotulo=""):
    """Antes de carregar um modelo: VRAM tem que estar no nivel de referencia."""
    if target is None:
        return None
    used = gpu_used()
    if used is None:
        return None
    t0 = time.time()
    if used - target > VRAM_TOL:
        print(f"   VRAM acima do baseline antes de {rotulo or 'carregar'}: "
              f"{gib(used)} vs {gib(target)}; esperando liberar...")
    while used - target > VRAM_TOL and time.time() - t0 < UNLOAD_TIMEOUT:
        time.sleep(2)
        used = gpu_used()
    if used - target > VRAM_TOL:
        pids = kill_llama_remnants()
        if pids:
            print(f"   matei llama-server remanescente: {pids}")
            time.sleep(3)
            used = gpu_used()
    if used - target > VRAM_TOL:
        print(f"   !! VRAM ainda {gib(used)} (referencia {gib(target)}): outro "
              f"processo esta usando a GPU; o bench vai dividir CPU+RAM")
    else:
        print(f"   GPU limpa: {gib(used)} de VRAM em uso (referencia {gib(target)})")
    return used


def physical_cores():
    pairs = set()
    try:
        with open("/proc/cpuinfo") as f:
            phys = core = None
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip() and phys is not None and core is not None:
                    pairs.add((phys, core))
                    phys = core = None
    except Exception:
        pass
    return len(pairs) or (os.cpu_count() or 4) // 2


def build_cmd(path, mode, args):
    threads = args.threads or physical_cores()
    # -np 1: um slot so. Varios slots multiplicam o KV cache e reduzem o numero
    # de camadas que cabem na GPU, degradando o bench sem necessidade.
    cmd = [LLAMA_SERVER, "-m", path, "--jinja", "-c", str(CTX), "-np", "1",
           "-lv", "4", "--host", "127.0.0.1", "--port", str(PORT)]
    if args.mlock:
        cmd += ["-lm", "mlock"]
    if mode == "cpu":
        cmd += ["-dev", "none", "-ngl", "0", "-t", str(threads),
                "-tb", str(min(threads * 2, os.cpu_count() or threads))]
    else:
        cmd += ["-dev", DEVICE, "-t", str(threads), "-ngl", str(args.ngl)]
        if args.fit:
            cmd += ["-fit", "on", "-fitt", str(args.fit_target)]
        else:
            cmd += ["-fit", "off"]
    return cmd


def start_server(path, sess, mode, args, log_path):
    """Sobe o llama-server e espera o /health. Devolve (proc, log_path)."""
    log = open(log_path, "w")
    cmd = build_cmd(path, mode, args)
    sess.cmd = cmd
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    t0 = time.time()
    try:
        for _ in range(900):
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server saiu (rc={proc.returncode}); "
                                   f"log: {log_path}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health",
                                            timeout=3) as r:
                    if r.status == 200:
                        sess.load_s = round(time.time() - t0, 1)
                        return proc, log_path
            except Exception:
                pass
            time.sleep(1)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        proc.wait()
        raise
    finally:
        log.close()
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    proc.wait()
    raise RuntimeError(f"timeout esperando o llama-server; log: {log_path}")


def offload_info(log_path):
    try:
        with open(log_path, errors="replace") as f:
            txt = f.read()
    except OSError:
        return {"camadas_gpu": None, "camadas_total": None, "linhas": []}
    m = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", txt)
    if m:
        gpu_layers, total = (int(m[-1][0]), int(m[-1][1]))
    else:
        layers = re.findall(r"offloading (\d+) repeating layers to the GPU", txt)
        gpu_layers, total = (int(layers[-1]) if layers else None, None)
    return {"camadas_gpu": gpu_layers, "camadas_total": total,
            "linhas": [l.strip() for l in txt.splitlines() if "offload" in l][-8:]}


# ---------------------------------- runner ----------------------------------
def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else 0.0


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


class Session:
    """Um modelo rodando sob um commit: pasta de samples + metadados."""

    def __init__(self, model, commit, commit_short, day, args):
        self.model = model
        self.commit = commit
        self.commit_short = commit_short
        self.day = day
        self.args = args
        self.dir = os.path.join(day, f"{slug(model)}-{commit_short}")
        if os.path.exists(self.dir):
            raise RuntimeError(f"pasta de samples ja existe: {self.dir}")
        os.makedirs(self.dir)
        self.backend = "?"
        self.ngl = "?"
        self.cmd = []
        self.load_s = None
        self.model_path = None
        self.model_bytes = None
        self.unload = None
        self.inicio = time.strftime("%FT%T")
        self.fim = None
        self.samples = 0

    def write_manifest(self, rows, offload, gpu):
        self.fim = time.strftime("%FT%T")
        man = {
            "commit": self.commit,
            "commit_short": self.commit_short,
            "data": os.path.basename(self.day),
            "modelo": self.model,
            "arquivo": self.model_path,
            "tamanho_bytes": self.model_bytes,
            "tamanho_gib": round(self.model_bytes / 1024 ** 3, 2) if self.model_bytes else None,
            "backend": self.backend,
            "ngl": self.ngl,
            "camadas_gpu": (offload or {}).get("camadas_gpu"),
            "camadas_total": (offload or {}).get("camadas_total"),
            "offload_log": (offload or {}).get("linhas", []),
            "llama_server": LLAMA_SERVER,
            "cmd": self.cmd,
            "ctx": CTX,
            "threads": self.args.threads or physical_cores(),
            "args_bench": vars(self.args),
            "inicio": self.inicio,
            "fim": self.fim,
            "load_s": self.load_s,
            "vram_baseline_bytes": gpu.baseline if gpu else None,
            "vram_pico_bytes": gpu.peak if gpu else None,
            "vram_final_bytes": gpu_used(),
            "unload": self.unload,
            "samples": self.samples,
            "resumo": [dict(zip(
                ["modelo", "tarefa", "acertos", "tempo_med_s", "tokens_med",
                 "tok_s", "estouros", "motivo_falha"], r)) for r in rows],
        }
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump(man, f, ensure_ascii=False, indent=1)
        linhas = [f"# {self.model} — commit {self.commit_short}", "",
                  f"- arquivo: `{os.path.basename(self.model_path or '?')}` "
                  f"({man['tamanho_gib']} GiB)",
                  f"- backend: **{self.backend}** (ngl={self.ngl}, "
                  f"camadas na GPU={man['camadas_gpu']}/{man['camadas_total']})",
                  f"- inicio: {self.inicio} | load: {self.load_s}s",
                  f"- VRAM: baseline {gib(gpu.baseline if gpu else None)} | "
                  f"pico {gib(gpu.peak if gpu else None)} | "
                  f"final {gib(man['vram_final_bytes'])}",
                  f"- unload: {'OK' if self.unload and not self.unload['problemas'] else 'PROBLEMA'}",
                  "", "| tarefa | acertos | tempo med (s) | tokens med | tok/s | "
                  "estouros | motivo mais comum |", "|---|---|---|---|---|---|---|"]
        linhas += [f"| {r[1]} | {r[2]} | {r[3]:.1f} | {r[4]:.0f} | {r[5]:.0f} "
                   f"| {r[6]} | {r[7]} |" for r in rows]
        with open(os.path.join(self.dir, "summary.md"), "w") as f:
            f.write("\n".join(linhas) + "\n")


def save_sample(sess, task, name, ok, reason, resp, extra=None):
    path = os.path.join(sess.dir, f"{name}.json")
    rec = {"commit": sess.commit, "commit_short": sess.commit_short,
           "data": os.path.basename(sess.day), "modelo": sess.model,
           "tarefa": task, "teste": name, "ok": ok, "motivo": reason,
           "backend": sess.backend, "ngl": sess.ngl,
           "ctx": CTX, "temp": sess.args.temp,
           "max_tokens": sess.args.max_tokens, "timeout_s": sess.args.timeout}
    if resp is not None:
        rec.update({
            "diag": resp.diag(),
            "tempo_s": round(resp.dt, 3),
            "tokens": resp.toks,
            "prompt_tokens": resp.ptoks,
            "tok_s": resp.gen_tps,
            "pp_tok_s": resp.pp_tps,
            "finish_reason": resp.finish,
            "prompt": resp.prompt,
            "request": resp.request,
            "resposta_final": resp.content,
            "reasoning": resp.reasoning,
            "tool_calls": resp.tool_calls,
            "resposta_bruta": resp.raw,
        })
    if extra:
        rec.update(extra)
    with open(path, "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    sess.samples += 1
    return path


def top_reason(reasons):
    if not reasons:
        return ""
    r, n = collections.Counter(x[:70] for x in reasons).most_common(1)[0]
    return f"{r} (x{n})"


def evaluate(sess, args):
    rows = []
    estado = {"label": ""}

    def ask(prompt, tools=None, max_tokens=None, label=None):
        rotulo = label or estado["label"] or "requisicao"
        live_r = live_c = None
        if args.stream:
            print(f"   → {rotulo} | prompt {len(prompt)} chars")
            live_r = Live("      ~ ", dim=True)
            live_c = Live("      > ")
        r = chat([{"role": "user", "content": prompt}], tools, args.temp,
                 max_tokens or args.max_tokens, args.timeout, args.stream,
                 live_r, live_c)
        r.prompt = prompt
        if args.stream:
            tps = f", {r.gen_tps:.0f} tok/s" if r.gen_tps else ""
            print(f"   ← {rotulo}: {r.toks} tok em {r.dt:.1f}s{tps}, "
                  f"finish={r.finish}")
        return r

    for tname, fn in TASKS:
        if args.tasks and not any(t in tname for t in args.tasks.split(",")):
            continue
        results, reasons = [], []
        for i in range(args.runs):
            name = f"{slug(tname)}-run{i + 1}"
            estado["label"] = f"{tname} run {i + 1}"
            try:
                ok, reason, resp = fn(i, ask)
            except TaskAbort as e:
                path = save_sample(sess, tname, name, False, str(e), None)
                print(f"   ! {tname} run {i + 1}: {e}  -> pulando tarefa\n"
                      f"       log: {path}")
                results.append((False, None))
                reasons.append(str(e))
                break
            except Exception as e:
                path = save_sample(sess, tname, name, False,
                                   f"erro no script: {e!r}", None)
                print(f"   ! {tname} run {i + 1}: erro no script: {e!r}\n"
                      f"       log: {path}")
                results.append((False, None))
                reasons.append(f"erro no script: {e!r}")
                break
            if ok is None:
                print(f"   - {tname}: pulada ({reason})")
                results = []
                break
            if not ok and resp is not None and resp.finish == "length":
                reason = (f"estourou max_tokens ({resp.toks} tok)"
                          + ("; resposta vazia (tudo no raciocinio?)"
                             if not resp.content.strip() else "; resposta cortada")
                          + f" [checagem: {reason}]")
            elif not ok and resp is not None and not resp.content.strip():
                reason = f"resposta vazia [checagem: {reason}]"
            path = save_sample(sess, tname, name, ok, reason, resp)
            results.append((ok, resp))
            if not ok:
                reasons.append(reason)
                print(f"   x {tname} run {i + 1}: {reason}\n"
                      f"       {resp.diag()}\n       log: {path}")
                if args.fail_fast:
                    print("       --fail-fast: pulando as repeticoes restantes")
                    break
        if not results:
            continue
        resps = [r for _, r in results if r is not None]
        n_ok = sum(1 for ok, _ in results if ok)
        overflow = sum(1 for r in resps if r.finish == "length")
        row = (sess.model, tname, f"{n_ok}/{len(results)}",
               med([r.dt for r in resps]), med([r.toks for r in resps]),
               med([r.gen_tps for r in resps]), overflow, top_reason(reasons))
        rows.append(row)
        warn = f"  [{overflow}x estouro de max_tokens]" if overflow else ""
        print(f"   {tname}: {row[2]}  ({row[3]:.1f}s, {row[4]:.0f} tok, "
              f"{row[5]:.0f} tok/s){warn}")

    if not args.tasks or "tool" in args.tasks:
        ok, total, resps, reasons, casos = t_tools(ask)
        for c in casos:
            save_sample(sess, "tool calling", f"tool-calling-{c['i']:02d}",
                        c["ok"], c["why"], c["resp"])
        save_sample(sess, "tool calling", "tool-calling-resumo", ok == total,
                    top_reason(reasons), None,
                    {"prompt": "20 casos de tool calling",
                     "acertos": ok, "total": total,
                     "respostas": total and round(ok / total, 3),
                     "falhas": [{"caso": c["i"], "prompt": c["prompt"],
                                 "motivo": c["why"]} for c in casos if not c["ok"]],
                     "ja_salvos_em": f"tool-calling-01..{total:02d}.json"})
        overflow = sum(1 for r in resps if r.finish == "length")
        rows.append((sess.model, "tool calling", f"{ok}/{total}",
                     med([r.dt for r in resps]), med([r.toks for r in resps]),
                     med([r.gen_tps for r in resps]), overflow, top_reason(reasons)))
        print(f"   tool calling: {ok}/{total}")
    return rows


def remote_size(url, timeout=30):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            n = int(r.headers.get("Content-Length") or 0)
            return n or None
    except Exception:
        return None


def download(url, path, expected=None):
    """Baixa/retoma e confere o tamanho final. True se o arquivo esta completo."""
    if subprocess.run(["curl", "-fL", "-#", "--retry", "3", "-C", "-",
                       "-o", path, url]).returncode != 0:
        return False
    got = os.path.getsize(path) if os.path.exists(path) else 0
    if expected and got != expected:
        print(f"   download incompleto: {got} de {expected} bytes")
        return False
    return True


def resolve_model(m, args):
    """Devolve (caminho, baixado_agora) ou (None, False)."""
    if "glob" in m:
        found = glob.glob(m["glob"])
        if not found:
            return None, False
        return found[0], False
    path = os.path.join(DL_DIR, m["file"])
    url = f"https://huggingface.co/{m['repo']}/resolve/main/{m['file']}"
    expected = remote_size(url)
    if os.path.exists(path):
        got = os.path.getsize(path)
        if expected is None or got == expected:
            return path, False
        print(f"   arquivo incompleto ({got} de {expected} bytes); retomando...")
    else:
        print("   baixando...")
    if not download(url, path, expected):
        return None, False
    return path, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="filtra modelos por substring")
    ap.add_argument("--tasks", default="", help="filtra tarefas por substring (ex: k8s,tool)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--timeout", type=int, default=240,
                    help="segundos sem resposta antes de abortar a tarefa")
    ap.add_argument("--fail-fast", action="store_true",
                    help="na 1a resposta errada, pula as repeticoes restantes da tarefa")
    ap.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="auto",
                    help="auto: GPU com sobra em CPU+RAM; gpu: so GPU; cpu: so CPU+RAM")
    ap.add_argument("--ngl", default="auto",
                    help="camadas na GPU: auto, all ou um numero (default: auto)")
    ap.add_argument("--no-fit", dest="fit", action="store_false",
                    help="desliga o -fit (ajuste automatico das camadas)")
    ap.add_argument("--fit-target", type=int, default=1024,
                    help="margem de VRAM livre em MiB para o -fit (default: 1024)")
    ap.add_argument("--threads", type=int, default=0,
                    help="threads de CPU (0 = nucleos fisicos)")
    ap.add_argument("--mlock", action="store_true",
                    help="trava o modelo na RAM (evita swap, exige ulimit -l)")
    ap.add_argument("--commit", default="",
                    help="reaproveita uma hash de commit (default: gera aleatoria)")
    ap.add_argument("--stream", dest="stream", action="store_true",
                    default=True,
                    help="mostra a resposta token a token no console (default)")
    ap.add_argument("--no-stream", dest="stream", action="store_false",
                    help="espera a resposta completa (sem saida ao vivo)")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    os.makedirs(DL_DIR, exist_ok=True)
    commit = args.commit or secrets.token_hex(20)
    commit_short = commit[:12]

    day = os.path.join(DETAILS_ROOT, time.strftime("%Y-%m-%d"))
    if os.path.isdir(day):
        print(f"pasta do dia existente: {day}")
    else:
        os.makedirs(day)
        print(f"pasta do dia criada: {day}")
    print(f"commit {commit_short} ({commit})")
    base_global = gpu_used()
    if base_global is not None:
        print(f"GPU antes do bench: {gib(base_global)} de {gib(gpu_total())} em uso")

    all_rows, dirs = [], []
    for m in MODELS:
        if args.only and args.only not in m["name"]:
            continue
        print(f"\n=== {m['name']}")
        path, downloaded = resolve_model(m, args)
        if not path:
            print("   arquivo nao encontrado, pulando")
            continue
        # Se baixamos/retomamos agora, o arquivo e descartado no fim (--keep mantem).
        downloaded_path = path if downloaded else None
        try:
            sess = Session(m["name"], commit, commit_short, day, args)
        except RuntimeError as e:
            print(f"   {e}")
            continue
        sess.model_path, sess.model_bytes = path, os.path.getsize(path)
        dirs.append(sess.dir)
        print(f"   samples: {sess.dir}  ({gib(sess.model_bytes)})")

        # Nada de outro modelo pode estar ocupando a GPU agora.
        if port_open():
            pids = kill_llama_remnants()
            print(f"   porta {PORT} ocupada; limpei llama-server: {pids}")
            t0 = time.time()
            while port_open() and time.time() - t0 < 20:
                time.sleep(0.3)
        ensure_gpu_clean(base_global, f"carregar {m['name']}")

        gpu = Gpu()
        gpu.start()
        if args.backend != "cpu" and gpu.baseline and gpu.baseline > 2 * 1024 ** 3:
            print(f"   aviso: {gib(gpu.baseline)} de VRAM ja em uso por outros "
                  f"apps (jogo/navegador?); o llama.cpp vai mandar o que nao "
                  f"couber para CPU+RAM e o tok/s cai")
        proc, log_path, offload, rows = None, None, None, []
        try:
            modes = ["cpu"] if args.backend == "cpu" else (
                ["gpu"] if args.backend == "gpu" else ["gpu", "cpu"])
            erros = []
            for mode in modes:
                try:
                    print(f"   subindo llama-server ({mode})...")
                    attempt_log = (f"/tmp/llama-{slug(sess.model)}-"
                                   f"{sess.commit_short}-{mode}.log")
                    proc, log_path = start_server(path, sess, mode, args,
                                                  attempt_log)
                    sess.backend = mode
                    break
                except Exception as e:
                    erros.append(f"{mode}: {e}")
                    print(f"   {mode} falhou: {e}")
                    if os.path.exists(attempt_log):
                        shutil.copy(attempt_log, os.path.join(
                            sess.dir, f"llama-server-{mode}-falhou.log"))
                    proc = None
            if proc is None:
                raise RuntimeError("; ".join(erros))

            offload = offload_info(log_path)
            sess.ngl = args.ngl if sess.backend == "gpu" else "0"
            camadas = (f"{offload['camadas_gpu']}/{offload['camadas_total']}"
                       if offload["camadas_gpu"] is not None else "n/d")
            print(f"   pronto em {sess.load_s}s | VRAM {gib(gpu_used())} "
                  f"(baseline {gib(gpu.baseline)}) | camadas na GPU={camadas}")
            rows = evaluate(sess, args)
            all_rows += rows
        except Exception as e:
            print(f"   falhou: {e}")
        finally:
            stop_and_verify(proc, sess, gpu)
            gpu.stop()
            if log_path and os.path.exists(log_path):
                shutil.copy(log_path, os.path.join(sess.dir, "llama-server.log"))
            sess.write_manifest(rows, offload, gpu)
            if downloaded_path and not args.keep:
                os.remove(downloaded_path)

    lines = ["| modelo | tarefa | acertos | tempo med (s) | tokens med | tok/s | "
             "estouros | motivo mais comum de falha |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.1f} | {r[4]:.0f} | {r[5]:.0f} | "
              f"{r[6]} | {r[7]} |" for r in all_rows]
    table = "\n".join(lines)
    print("\n" + table)
    with open(OUT_MD, "a") as f:
        f.write(f"\n### {time.strftime('%F %H:%M')} (commit {commit_short}, "
                f"temp={args.temp}, runs={args.runs}, "
                f"max_tokens={args.max_tokens}, backend={args.backend})\n\n"
                + table + "\n")

    with open(os.path.join(day, f"sessao-{commit_short}.json"), "w") as f:
        json.dump({"commit": commit, "commit_short": commit_short,
                   "inicio": time.strftime("%FT%T"), "args": vars(args),
                   "pastas": dirs,
                   "resumo": [dict(zip(
                       ["modelo", "tarefa", "acertos", "tempo_med_s",
                        "tokens_med", "tok_s", "estouros", "motivo_falha"], r))
                       for r in all_rows]}, f, ensure_ascii=False, indent=1)

    print(f"\nsalvo em {OUT_MD}\nsamples em {day}")


if __name__ == "__main__":
    main()
