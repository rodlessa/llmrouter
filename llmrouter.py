"""
llmrouter - proxy OpenAI-compativel: llama.cpp local para tarefas simples,
API na nuvem para tarefas pesadas (tools, busca web, contexto grande, agentes).

Instalacao rapida:
    ./install.sh                 # venv + envs + units systemd --user
    ./install.sh --with-laya     # + decisor local (torch CPU + laya)

Manual:
    uv venv && uv pip install -r requirements.txt
    cp router.env.example router.env    # edite API_KEY/API_BASE_URL/API_MODEL
    cp local.env.example  local.env     # edite LLAMA_SERVER/MODEL_PATH
    set -a; . ./router.env; set +a
    .venv/bin/uvicorn llmrouter:app --host 127.0.0.1 --port 4000

No OpenClaw, aponte um provider para http://127.0.0.1:4000/v1 e use os modelos
"auto" (decide), "local" ou "api" (forcam o destino).

Rotas, em ordem:
  1. modelo forcado (local/api)
  2. loop de tools em andamento        -> api
  3. contexto acima de 60% da janela   -> api
  4. tools fora da allowlist           -> api
  5. palavra-chave de tarefa pesada    -> api
  6. resto                             -> local
O Laya (opcional) pode opinar entre 4 e 6 conforme LAYA_MODE.

Ciclo de vida do llama-server: sobe sob demanda, cai depois de
LOCAL_IDLE_STOP_SECONDS ocioso, e nao sobe se a VRAM livre estiver abaixo de
LOCAL_MIN_FREE_VRAM_MIB.

Toda resposta de agente recebe um rodape ASCII no fim dizendo onde rodou.
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import threading
import time

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8080/v1")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "local")
LOCAL_CTX = int(os.getenv("LOCAL_CTX", "16384"))  # mesmo valor do -c do llama-server
LOCAL_CTX_USABLE = 0.6  # acima disso de contexto, manda para a API

API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("API_KEY", "")
API_MODEL = os.getenv("API_MODEL", "deepseek-v4-flash")
# Modelo de "pensamento" (planejamento, arquitetura, review). Vazio = usa so API_MODEL.
API_MODEL_THINK = os.getenv("API_MODEL_THINK", "")

# Tools simples e seguras que o modelo local pode tentar (vazio = nenhuma)
LOCAL_TOOL_ALLOWLIST = {
    t.strip() for t in os.getenv("LOCAL_TOOL_ALLOWLIST", "").split(",") if t.strip()
}
# "*" = o modelo local pode receber o conjunto completo de tools do cliente.
LOCAL_ALLOW_ALL_TOOLS = "*" in LOCAL_TOOL_ALLOWLIST

# ------------------------------- Laya (local) -------------------------------
# System One model Apache-2.0 que roda em CPU: state + perguntas tipadas ->
# escolha com probabilidades, ~100-200 ms, sem custo.
# off     = nao roda
# advise  = roda, registra em decisions.jsonl, mas NAO muda a rota
# decide  = usa a escolha do Laya quando a probabilidade passa de LAYA_DECIDE_P
LAYA_MODE = os.getenv("LAYA_MODE", "advise")
LAYA_DECIDE_P = float(os.getenv("LAYA_DECIDE_P", "0.75"))
LAYA_LOG = os.path.expanduser(
    os.getenv("LAYA_LOG", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "decisions.jsonl")))

# ------------------------------- rodape -------------------------------------
FOOTER_ENABLED = os.getenv("FOOTER", "1").lower() not in ("0", "false", "no")
FOOTER_REQUIRE_TOOLS = os.getenv("FOOTER_REQUIRE_TOOLS", "1") != "0"

# Sinais de tarefa de pensamento: vale pagar o modelo mais caro.
THINK_PATTERNS = re.compile(
    r"(planej|plane?jamento|arquitet|architect|\bdesign\b|estrat[eé]g|strateg|"
    r"trade-?off|roadmap|decis[ãa]o|decidir|avalie|avaliar|compar[ae]|"
    r"analis[ae]|an[áa]lise|\breview\b|revis[ãa]o|critique|abordagem|approach|"
    r"boas pr[áa]ticas|best practice|estrutur|organiz[ae]|refator|refactor|"
    r"como voc[êe] (faria|implementaria|estruturaria)|proposta)",
    re.IGNORECASE,
)

HEAVY_PATTERNS = re.compile(
    r"(pesquis[ae]|busca(r)? na (web|internet)|web ?search|google|"
    r"agent|sub-?agent|refator|arquitetura|architecture|"
    r"projeto|project|repositor|repo\b|debug|stack ?trace|"
    r"analis[ae] (o )?c[oó]digo|code review|plano|plan\b|"
    r"terraform|kubernetes|k8s|migra[cç][aã]o|migration)",
    re.IGNORECASE,
)

app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0))

# ---------------------- ciclo de vida do modelo local -----------------------
# O llama-server nao fica carregado a toa: sobe sob demanda e cai sozinho
# depois de N segundos sem requisicao (libera VRAM para jogos).
LOCAL_SERVICE = os.getenv("LOCAL_SERVICE", "llama-local.service")
LOCAL_AUTOSTART = os.getenv("LOCAL_AUTOSTART", "1").lower() not in ("0", "false", "no")
LOCAL_IDLE_STOP = int(os.getenv("LOCAL_IDLE_STOP_SECONDS", "300"))
LOCAL_MIN_FREE_MIB = int(os.getenv("LOCAL_MIN_FREE_VRAM_MIB", "13500"))
ROCM_SMI = shutil.which("rocm-smi")
SYSTEMCTL = shutil.which("systemctl")
LOCAL_HEALTH_URL = LOCAL_BASE_URL.rsplit("/v1", 1)[0] + "/health"

STATE = {"active": 0, "last": time.time(), "rotas": {}}

# Headers que alguns clientes usam para identificar a sessao. Sem nenhum deles,
# o aviso de mudanca de rota vale para o processo inteiro.
SID_HEADERS = ("x-session-id", "x-openclaw-session", "x-openclaw-session-key",
               "x-request-session", "x-conversation-id")


def session_key(request):
    for h in SID_HEADERS:
        v = request.headers.get(h)
        if v:
            return v
    return "global"


def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        print(f"[router] erro ao rodar {cmd}: {e}", flush=True)
        return None


def local_healthy():
    try:
        return httpx.get(LOCAL_HEALTH_URL, timeout=3).status_code == 200
    except Exception:
        return False


def local_service_active():
    if not SYSTEMCTL:
        return False
    r = sh([SYSTEMCTL, "--user", "is-active", LOCAL_SERVICE])
    return bool(r and r.stdout.strip() == "active")


def vram_free_mib(card=0):
    if not ROCM_SMI:
        return None
    r = sh([ROCM_SMI, "--showmeminfo", "vram", "--json"], 20)
    try:
        d = json.loads(r.stdout)[f"card{card}"]
        total = int(d["VRAM Total Memory (B)"])
        used = int(d["VRAM Total Used Memory (B)"])
        return (total - used) // (1024 * 1024)
    except Exception:
        return None


async def ensure_local_ready():
    """Garante o llama-server no ar (subindo sob demanda). (ok, motivo)."""
    if await asyncio.to_thread(local_healthy):
        return True, ""

    # Servico ativo mas ainda carregando o modelo: so espera.
    if await asyncio.to_thread(local_service_active):
        return await asyncio.to_thread(wait_health, 240), "carregando"

    if not LOCAL_AUTOSTART:
        return False, "autostart desligado"
    if not SYSTEMCTL:
        return False, "systemctl indisponivel"

    free = await asyncio.to_thread(vram_free_mib)
    if free is not None and free < LOCAL_MIN_FREE_MIB:
        return False, f"VRAM insuficiente ({free} MiB livres)"

    t0 = time.time()
    print(f"[router] subindo {LOCAL_SERVICE} sob demanda...", flush=True)
    r = await asyncio.to_thread(
        sh, [SYSTEMCTL, "--user", "start", LOCAL_SERVICE], 120)
    if r is None or r.returncode != 0:
        rc = "sem resposta" if r is None else f"rc={r.returncode}"
        return False, f"systemctl start falhou ({rc})"
    ok = await asyncio.to_thread(wait_health, 240)
    print(f"[router] {LOCAL_SERVICE} pronto em {time.time() - t0:.0f}s"
          if ok else f"[router] {LOCAL_SERVICE} nao subiu", flush=True)
    return ok, "" if ok else "nao subiu a tempo"


def wait_health(timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if local_healthy():
            return True
        time.sleep(2)
    return False


async def idle_watcher():
    """Para o llama-server quando fica ocioso, devolvendo a VRAM."""
    while True:
        await asyncio.sleep(15)
        if LOCAL_IDLE_STOP <= 0 or STATE["active"] > 0:
            continue
        if time.time() - STATE["last"] < LOCAL_IDLE_STOP:
            continue
        if await asyncio.to_thread(local_service_active):
            await asyncio.to_thread(
                sh, [SYSTEMCTL, "--user", "stop", LOCAL_SERVICE], 90)
            print(f"[router] {LOCAL_SERVICE} parado por inatividade "
                  f"({LOCAL_IDLE_STOP}s)", flush=True)


@app.on_event("startup")
async def _startup():
    asyncio.create_task(idle_watcher())


def _text_of(msg: dict) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # content em partes
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def estimate_tokens(body: dict) -> int:
    chars = sum(len(_text_of(m)) for m in body.get("messages", []))
    chars += len(json.dumps(body.get("tools", [])))
    return chars // 4  # estimativa grosseira


def tool_names(body: dict) -> set:
    return {
        t.get("function", {}).get("name", "")
        for t in body.get("tools", []) or []
    }


def decide(body: dict) -> tuple[str, str, bool]:
    """Retorna (destino, motivo, hard). hard=True nao pode ser revisto pelo Jev."""
    model = body.get("model", "")
    # OpenClaw manda so o id do modelo (parte depois da "/"), entao aceita
    # tanto "router/local" quanto "local".
    tail = model.rsplit("/", 1)[-1]
    if tail == "local":
        if estimate_tokens(body) > LOCAL_CTX * LOCAL_CTX_USABLE:
            return "api", "local forcado, mas contexto grande", True
        return "local", "forcado", True
    if tail == "api":
        return "api", "forcado", True

    msgs = body.get("messages", [])

    # 1) Meio de um loop de tools: não troque de modelo no meio
    if any(m.get("role") == "tool" for m in msgs):
        return "api", "loop de tools em andamento", True

    # 2) Tools exigem tool-calling confiável; local só se tudo estiver na allowlist
    names = tool_names(body)
    if names and not LOCAL_ALLOW_ALL_TOOLS and not names.issubset(LOCAL_TOOL_ALLOWLIST):
        return "api", "tools fora da allowlist", False

    # 3) Contexto grande demais para o modelo local
    if estimate_tokens(body) > LOCAL_CTX * LOCAL_CTX_USABLE:
        return "api", "contexto grande", True

    # 4) Palavras-chave de tarefa pesada na última mensagem do usuário
    last_user = next(
        (_text_of(m) for m in reversed(msgs) if m.get("role") == "user"), ""
    )
    if HEAVY_PATTERNS.search(last_user):
        return "api", "palavra-chave de tarefa pesada", False

    return "local", "tarefa simples", False


def _last_user(body):
    msgs = body.get("messages", [])
    return next((_text_of(m) for m in reversed(msgs) if m.get("role") == "user"), "")


_laya = None
_laya_lock = threading.Lock()


def laya_predict_sync(body):
    """Roda o Laya em CPU (carrega o checkpoint no primeiro uso)."""
    global _laya
    with _laya_lock:
        if _laya is None:
            from laya import Router
            _laya = Router()
    state = (
        f"User request:\n{_last_user(body)[:1200]}\n\n"
        f"Conversation messages: {len(body.get('messages', []))}\n"
        f"Estimated prompt tokens: {estimate_tokens(body)} (local window: {LOCAL_CTX})\n"
    )
    questions = {"executor": {
        "type": "choice",
        "instructions": "Which executor should answer this request?",
        "criteria": {
            "local": "short self-contained answer: chat, definition, explanation, "
                     "translation, summary, small self-contained code snippet",
            "api": "needs tool execution, codebase work, deep reasoning or long context",
        },
    }}
    ans = _laya.predict(state, questions)["answers"]["executor"]
    return {"escolha": ans.get("choice"),
            "probabilidades": ans.get("probabilities") or {}}


async def laya_route(body):
    return await asyncio.to_thread(laya_predict_sync, body)


def log_decision(rec):
    """Append-only: base para fine-tunar o Laya no nosso dominio."""
    if not LAYA_LOG:
        return
    try:
        with open(LAYA_LOG, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[router] nao consegui gravar {LAYA_LOG}: {e}", flush=True)


def footer_box(target, usage, timings, elapsed, mudou=None, api_model=None):
    """Caixa ASCII dizendo onde a resposta rodou. Sem emoji."""
    if target == "local":
        titulo = "ROTA: LOCAL (llama.cpp)"
        tok = usage.get("completion_tokens") or timings.get("predicted_n")
        ptok = usage.get("prompt_tokens") or timings.get("prompt_n")
        tps = timings.get("predicted_per_second")
    else:
        titulo = f"ROTA: API ({api_model or API_MODEL})"
        tok = usage.get("completion_tokens")
        ptok = usage.get("prompt_tokens")
        tps = None
    detalhes = []
    if ptok:
        detalhes.append(f"{ptok} prompt tok")
    if tok:
        detalhes.append(f"{tok} gerados")
    if tps:
        detalhes.append(f"{tps:.0f} tok/s")
    detalhes.append(f"{elapsed:.1f}s")
    linhas = [titulo, " | ".join(detalhes)]
    if mudou:
        linhas.append(f"mudou: {mudou}")
    larg = max(len(x) for x in linhas)
    borda = "+" + "-" * (larg + 2) + "+"
    corpo = "\n".join(f"| {x.ljust(larg)} |" for x in linhas)
    return f"\n\n{borda}\n{corpo}\n{borda}"


async def relay_stream(resp, footer_fn):
    """Repassa o SSE e injeta o rodape antes do finish_reason."""
    buf = b""
    pending = None
    tail = []
    usage, timings = {}, {}
    meta = {"id": None, "model": None, "created": None}
    saw_content = False

    def sse(obj):
        return ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()

    async for chunk in resp.aiter_bytes():
        buf += chunk
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            line = raw.decode("utf-8", "replace").rstrip("\r")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except ValueError:
                yield (line + "\n\n").encode()
                continue
            for k in ("id", "model", "created"):
                if obj.get(k) is not None:
                    meta[k] = obj[k]
            if obj.get("usage"):
                usage.update(obj["usage"])
            if obj.get("timings"):
                timings.update(obj["timings"])
            ch = (obj.get("choices") or [None])[0]
            if ch:
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    saw_content = True
                if ch.get("finish_reason") and pending is None:
                    pending = obj
                    continue
            if pending is not None:
                tail.append(obj)
            else:
                yield (line + "\n\n").encode()

    if footer_fn and saw_content:
        extra = {"id": meta["id"], "object": "chat.completion.chunk",
                 "created": meta["created"], "model": meta["model"],
                 "choices": [{"index": 0,
                              "delta": {"content": footer_fn(usage, timings)},
                              "finish_reason": None}]}
        yield sse(extra)
    if pending is not None:
        yield sse(pending)
    for obj in tail:
        yield sse(obj)
    yield b"data: [DONE]\n\n"


def tipo_erro(e):
    """Descricao curta do erro, sem URL/chave."""
    if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
        return f"HTTP {e.response.status_code}"
    if isinstance(e, httpx.ConnectError):
        return "conexao recusada"
    if isinstance(e, httpx.TimeoutException):
        return "timeout"
    return type(e).__name__


def api_model_for(body: dict):
    """Escolhe o modelo da API: pro para pensar, flash para executar."""
    if not API_MODEL_THINK or API_MODEL_THINK == API_MODEL:
        return API_MODEL, ""
    msgs = body.get("messages", [])
    if any(m.get("role") == "tool" or m.get("tool_calls") for m in msgs):
        return API_MODEL, "exec (loop de tools)"
    if THINK_PATTERNS.search(_last_user(body)):
        return API_MODEL_THINK, "pensamento"
    return API_MODEL, "execucao"


def upstream(target: str, api_model=None) -> tuple[str, dict, str]:
    if target == "local":
        return LOCAL_BASE_URL, {}, LOCAL_MODEL
    return API_BASE_URL, {"Authorization": f"Bearer {API_KEY}"}, (api_model or API_MODEL)


async def forward(body: dict, target: str, api_model=None):
    """Abre a conexão e devolve a response em streaming do httpx."""
    base, headers, model = upstream(target, api_model)
    payload = {**body, "model": model}
    req = client.build_request(
        "POST", f"{base}/chat/completions", json=payload, headers=headers
    )
    return await client.send(req, stream=True)


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    started = time.time()
    target, reason, hard = decide(body)

    laya = None
    if LAYA_MODE != "off" and not hard:
        try:
            laya = await laya_route(body)
        except Exception as e:
            print(f"[router] laya falhou: {type(e).__name__}: {e}", flush=True)
        if laya and LAYA_MODE == "decide":
            esc = laya["escolha"]
            prob = (laya["probabilidades"] or {}).get(esc, 0.0)
            names = tool_names(body)
            pode_local = (LOCAL_ALLOW_ALL_TOOLS or not names
                          or names.issubset(LOCAL_TOOL_ALLOWLIST))
            if (esc in ("local", "api") and prob >= LAYA_DECIDE_P
                    and not (esc == "local" and not pode_local)):
                if esc != target:
                    reason = f"laya:{esc} p={prob:.2f} (heuristica dizia {target})"
                target = esc
        elif laya:
            print(f"[router] laya p_local="
                  f"{(laya['probabilidades'] or {}).get('local', 0):.2f} "
                  f"(rota {target})", flush=True)

    if os.getenv("ROUTER_DEBUG"):
        print(f"[router] model={body.get('model')!r} "
              f"est_tok={estimate_tokens(body)} "
              f"tools={sorted(t for t in tool_names(body) if t)}", flush=True)

    api_model, api_porque = api_model_for(body)
    add_footer = FOOTER_ENABLED and (
        not FOOTER_REQUIRE_TOOLS or bool(tool_names(body)))

    if target == "local":
        ok, why = await ensure_local_ready()
        if not ok:
            target, reason = "api", f"local indisponivel ({why})"

    STATE["active"] += 1
    released = False

    def done():
        nonlocal released
        if released:
            return
        released = True
        STATE["active"] = max(0, STATE["active"] - 1)
        STATE["last"] = time.time()

    fallback = False
    try:
        resp = await forward(body, target, api_model)
        if target == "local" and resp.status_code >= 500:
            await resp.aclose()
            raise httpx.HTTPError("local 5xx")
    except httpx.HTTPError as e:
        if target == "api":
            done()
            return JSONResponse(
                {"error": "API upstream indisponivel", "detail": tipo_erro(e)}, 502)
        # Local caiu/timeout: escala para a API
        target, reason, fallback = "api", "fallback: local falhou", True
        try:
            resp = await forward(body, target, api_model)
        except httpx.HTTPError as e:
            done()
            return JSONResponse(
                {"error": "API upstream indisponivel", "detail": tipo_erro(e)}, 502)

    if target == "api" and api_porque:
        reason = f"{reason}; {api_model} ({api_porque})"
    print(f"[router] -> {target} ({reason}){' [FALLBACK]' if fallback else ''}",
          flush=True)
    log_decision({"ts": round(time.time(), 3), "est_tok": estimate_tokens(body),
                  "n_msgs": len(body.get("messages", [])),
                  "tools": sorted(t for t in tool_names(body) if t),
                  "last_user": _last_user(body)[:500],
                  "hard": hard, "laya": laya, "final": target, "motivo": reason})

    out_headers = {"X-Routed-To": target, "X-Route-Reason": reason}

    # Avisa no rodape quando a rota muda em relacao ao request anterior.
    mudou = None
    if add_footer:
        sid = session_key(request)
        anterior = STATE["rotas"].get(sid)
        if anterior and anterior != target:
            mudou = f"{anterior.upper()} -> {target.upper()}"
            print(f"[router] rota mudou ({sid}): {anterior} -> {target}",
                  flush=True)
        STATE["rotas"][sid] = target

    def inject_footer(usage, timings):
        return footer_box(target, usage, timings, time.time() - started, mudou,
                          api_model if target == "api" else None)

    if not body.get("stream"):
        raw = await resp.aread()
        await resp.aclose()
        done()
        ctype = resp.headers.get("content-type", "application/json")
        try:
            data = json.loads(raw)
        except ValueError:
            return Response(content=raw, media_type=ctype, headers=out_headers)
        if add_footer:
            try:
                msg = data["choices"][0]["message"]
                if msg.get("content"):
                    msg["content"] += footer_box(
                        target, data.get("usage") or {},
                        data.get("timings") or {}, time.time() - started,
                        mudou, api_model if target == "api" else None)
            except Exception:
                pass
        return JSONResponse(content=data, headers=out_headers)

    async def body_iter():
        try:
            async for chunk in relay_stream(
                    resp, inject_footer if add_footer else None):
                yield chunk
        finally:
            await resp.aclose()
            done()

    return StreamingResponse(
        body_iter(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
        headers=out_headers,
    )


@app.get("/local/status")
async def local_status():
    return {
        "servico": LOCAL_SERVICE,
        "ativo": await asyncio.to_thread(local_service_active),
        "saudavel": await asyncio.to_thread(local_healthy),
        "requisicoes_em_voo": STATE["active"],
        "ocioso_s": round(time.time() - STATE["last"], 1),
        "autostart": LOCAL_AUTOSTART,
        "idle_stop_s": LOCAL_IDLE_STOP,
        "vram_livre_mib": await asyncio.to_thread(vram_free_mib),
        "api": {"exec": API_MODEL, "think": API_MODEL_THINK or None},
        "laya": {"modo": LAYA_MODE, "carregado": _laya is not None,
                 "decide_p": LAYA_DECIDE_P},
        "rodape": FOOTER_ENABLED,
        "rotas_sessao": STATE["rotas"],
    }


@app.post("/route/explain")
async def route_explain(request: Request):
    """Mostra a decisao sem encaminhar nada (util para calibrar)."""
    body = await request.json()
    target, reason, hard = decide(body)
    out = {"heuristica": {"destino": target, "motivo": reason, "hard": hard},
           "api_model": dict(zip(("modelo", "porque"), api_model_for(body)))}
    if LAYA_MODE != "off":
        try:
            out["laya"] = await laya_route(body)
            out["laya"]["modo"] = LAYA_MODE
        except Exception as e:
            out["laya"] = {"erro": f"{type(e).__name__}: {e}"}
    else:
        out["laya"] = {"ligado": False}
    return out


@app.get("/v1/models")
async def models():
    ids = ["router/auto", "router/local", "router/api"]
    return {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}
