#!/usr/bin/env python3
"""
run_suite.py - benchmark agnostico de modelos GGUF locais.

Le a suite de um JSON (modelos + tarefas + checkers) e mede acerto e tempo por
categoria. Nada de modelo ou tarefa fica no codigo: quem quiser testar outro
modelo ou outra tarefa edita o suite.json.

Uso:
    python3 run_suite.py                       # usa suite.json ao lado
    python3 run_suite.py --suite minha.json
    python3 run_suite.py --models gpt-oss --tasks code_review --runs 3
    python3 run_suite.py --ctx 65536           # testa janela maior

Saida:
    ~/quality-details/<data>/<commit>/<modelo>/<tarefa>-runN.json   (samples)
    ~/quality-details/<data>/<commit>/results.json                  (bruto)
    ~/quality-details/<data>/<commit>/summary.md                    (tabela)

Checkers disponiveis (campo task.check.type):
    python_tests    roda o bloco ```python + asserts
    python_files    blocos com "# file: nome.py" importados e comparados
    json_groups     JSON de resposta; exige >=1 termo de cada grupo
    contains_groups exige >=1 termo de cada grupo no texto
    contains_all    exige todos os termos de check.terms
    tool_cases      tool calling (get_weather / run_sql)
"""
import argparse
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
import time
import urllib.error
import urllib.request

LLAMA_SERVER = os.path.expanduser(
    os.getenv("LLAMA_SERVER", "~/ai/llama.cpp/build/bin/llama-server"))
DEVICE = os.getenv("BENCH_DEVICE", "Vulkan0")
DL_DIR = os.path.expanduser(os.getenv("BENCH_DL_DIR", "~/models/tmp"))
OUT_ROOT = os.path.expanduser(os.getenv("BENCH_OUT", "~/quality-details"))
HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------- infraestrutura
class Abort(Exception):
    """Erro de infra: aborta a tarefa/modelo."""


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else 0.0


def gib(n):
    return "n/d" if n is None else f"{n / 1024 ** 3:.2f} GiB"


def port_open(port):
    s = socket.socket()
    s.settimeout(0.4)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def remote_size(url, timeout=30):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return int(r.headers.get("Content-Length") or 0) or None
    except Exception:
        return None


def model_path(m):
    """Caminho local do .gguf, baixando da HF se preciso."""
    if m.get("path"):
        p = os.path.expanduser(m["path"])
        if not os.path.exists(p):
            raise Abort(f"arquivo nao encontrado: {p}")
        return p
    repo, fname = m.get("hf_repo"), m.get("hf_file")
    if not (repo and fname):
        raise Abort("modelo precisa de path ou hf_repo+hf_file")
    os.makedirs(DL_DIR, exist_ok=True)
    path = os.path.join(DL_DIR, fname)
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    expected = remote_size(url)
    if os.path.exists(path):
        got = os.path.getsize(path)
        if expected is None or got == expected:
            return path
        print(f"   arquivo incompleto ({got}/{expected}), retomando...")
    else:
        print(f"   baixando {fname}...")
    rc = subprocess.run(["curl", "-fL", "-#", "--retry", "3", "-C", "-",
                         "-o", path, url]).returncode
    if rc != 0:
        raise Abort("download falhou")
    if expected and os.path.getsize(path) != expected:
        raise Abort("download incompleto")
    return path


def vram_used():
    exe = shutil.which("rocm-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--showmeminfo", "vram", "--json"],
                             capture_output=True, text=True, timeout=20).stdout
        return int(json.loads(out)["card0"]["VRAM Total Used Memory (B)"])
    except Exception:
        return None


def start_server(path, name, ctx, port):
    log_path = f"/tmp/bench-{slug(name)}-{port}.log"
    log = open(log_path, "w")
    cmd = [LLAMA_SERVER, "-m", path, "--jinja", "-c", str(ctx), "-np", "1",
           "-dev", DEVICE, "-ngl", "auto", "-fit", "on", "-fitt", "1024",
           "-lv", "3", "--host", "127.0.0.1", "--port", str(port)]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    log.close()
    t0 = time.time()
    for _ in range(900):
        if proc.poll() is not None:
            tail = open(log_path, errors="replace").read()[-800:]
            raise Abort(f"llama-server saiu (rc={proc.returncode}): {tail}")
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return proc, round(time.time() - t0, 1)
        except Exception:
            pass
        time.sleep(1)
    raise Abort("timeout subindo o llama-server")


def stop_server(proc, baseline, timeout=90):
    info = {"ok": True, "notas": []}
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=15)
            info["notas"].append("SIGKILL")
    t0 = time.time()
    while port_open(port_of(proc)) and time.time() - t0 < 15:
        time.sleep(0.25)
    used = vram_used()
    if baseline is not None and used is not None:
        deadline = time.time() + timeout
        while used - baseline > 1024 ** 3 and time.time() < deadline:
            time.sleep(2)
            used = vram_used()
        if used - baseline > 1024 ** 3:
            info["ok"] = False
            info["notas"].append(f"VRAM {gib(used)} vs baseline {gib(baseline)}")
    info["vram"] = used
    return info


_PORTS = {}


def port_of(proc):
    return _PORTS.get(id(proc), 8099)


# ----------------------------------------------------------------- chamada do modelo
def chat(port, prompt, tools=None, temp=0.2, max_tokens=3000, timeout=300):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}],
            "temperature": temp, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        raise Abort(f"HTTP {e.code}")
    except (socket.timeout, TimeoutError):
        raise Abort(f"timeout {timeout}s")
    except urllib.error.URLError as e:
        raise Abort(f"conexao: {e.reason}")
    msg = data["choices"][0]["message"]
    u = data.get("usage") or {}
    t = data.get("timings") or {}
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish": data["choices"][0].get("finish_reason"),
        "tokens": u.get("completion_tokens") or t.get("predicted_n") or 0,
        "ptokens": u.get("prompt_tokens") or t.get("prompt_n") or 0,
        "tok_s": t.get("predicted_per_second"),
        "dt": time.time() - t0,
        "request": body,
        "raw": data,
    }


# ----------------------------------------------------------------- checkers
def fenced(text):
    return re.findall(r"```[\w+-]*\n(.*?)```", text or "", re.S)


def first_block(text):
    b = fenced(text)
    return (b[0], True) if b else (text or "", False)


def run_py(code, tests="", timeout=20, cwd=None):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=cwd) as f:
        f.write(code + "\n" + (tests or ""))
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout, cwd=cwd)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    finally:
        os.unlink(path)


def flatten(obj):
    if isinstance(obj, dict):
        return " ".join(flatten(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(flatten(v) for v in obj)
    return str(obj)


def check_python_tests(task, resp):
    code, had = first_block(resp["content"])
    rc, _, err = run_py(code, task["check"].get("tests", ""))
    if rc == 0:
        return True, ""
    why = "timeout (loop infinito?)" if rc == 124 else \
        "testes falharam: " + (err.strip().splitlines() or ["?"])[-1][:160]
    return False, ("" if had else "sem bloco ```; ") + why


def check_python_files(task, resp):
    files = {}
    for blk in fenced(resp["content"]):
        m = re.match(r"\s*#\s*file:\s*(\S+)", blk)
        if m:
            files[m.group(1)] = blk
    missing = {"common.py", "a.py", "b.py"} - set(files)
    if missing:
        return False, f"arquivos ausentes: {sorted(missing)}"
    if ".replace(" in files["a.py"] or ".replace(" in files["b.py"]:
        return False, "logica duplicada mantida em a.py/b.py"
    # referencia: comportamento original
    ns = {}
    exec(open(os.path.join(HERE, "_ref_slugify.py")).read(), ns) \
        if os.path.exists(os.path.join(HERE, "_ref_slugify.py")) else None
    orig_a = ('def titulo_a(txt):\n    s = txt.strip().lower()\n'
              '    s = "".join(c if c.isalnum() else "-" for c in s)\n'
              '    while "--" in s:\n        s = s.replace("--", "-")\n'
              '    return s.strip("-")\n')
    orig_b = ('def titulo_b(txt, prefixo="post"):\n    s = txt.strip().lower()\n'
              '    s = "".join(c if c.isalnum() else "-" for c in s)\n'
              '    while "--" in s:\n        s = s.replace("--", "-")\n'
              '    return prefixo + "/" + s.strip("-")\n')
    ns2 = {}
    exec(orig_a + orig_b, ns2)
    ins = task["check"]["inputs"]
    expected = [[ns2["titulo_a"](s), ns2["titulo_b"](s), ns2["titulo_b"](s, "x")]
                for s in ins]
    with tempfile.TemporaryDirectory() as d:
        for name, content in files.items():
            open(os.path.join(d, os.path.basename(name)), "w").write(content)
        probe = ("import json\nfrom a import titulo_a\nfrom b import titulo_b\n"
                 f"ins = {ins!r}\n"
                 "print(json.dumps([[titulo_a(s), titulo_b(s), titulo_b(s, 'x')]"
                 " for s in ins]))\n")
        rc, out, err = run_py("", probe, cwd=d)
    if rc != 0:
        return False, "erro ao importar: " + (err.strip().splitlines() or ["?"])[-1][:120]
    try:
        got = json.loads(out)
    except ValueError:
        return False, "saida inesperada"
    if got != expected:
        for i, g, e in zip(ins, got, expected):
            if g != e:
                return False, f"comportamento mudou em {i!r}: {g} != {e}"
    return True, ""


def check_groups(text, groups):
    low = text.lower()
    missing = [g[0] for g in groups if not any(term.lower() in low for term in g)]
    return (not missing), ("faltou: " + ", ".join(f"[{m}]" for m in missing)
                           if missing else "")


def check_json_groups(task, resp):
    txt = resp["content"].strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return False, "sem JSON na resposta"
    try:
        obj = json.loads(m.group(0))
    except ValueError as e:
        return False, f"JSON invalido: {e}"
    flat = flatten(obj)
    ok, why = check_groups(flat, task["check"]["groups"])
    n = len(obj.get("issues") or []) if isinstance(obj, dict) else 0
    return ok, why + (f"; {n} issues" if ok else f"; {n} issues")


def check_contains_groups(task, resp):
    txt = resp["content"]
    ok, why = check_groups(txt, task["check"]["groups"])
    limite = task["check"].get("max_words")
    if ok and limite and len(txt.split()) > limite:
        return False, f"resposta longa demais ({len(txt.split())} palavras)"
    return ok, why


TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Clima atual de uma cidade",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "run_sql", "description": "Executa uma query SQL somente leitura",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]

CITIES = ["Lisboa", "Tokyo", "Fortaleza", "Berlin", "Oslo",
          "Nairobi", "Lima", "Sydney", "Toronto", "Cairo"]
MONTHS = ["janeiro", "fevereiro", "marco", "abril", "maio",
          "junho", "julho", "agosto", "setembro", "outubro"]


def longctx_prompt(seed=42, n=300):
    rng = random.Random(seed)
    vals, lines = {}, []
    for i in range(n):
        v = rng.randint(10000, 99999)
        vals[i] = v
        lines.append(f"def calc_tarifa_{i:03d}(x):\n"
                     f"    return x * {rng.randint(2, 9)} + {v}\n")
    target = 187
    prompt = ("Codigo:\n\n" + "\n".join(lines) +
              f"\n\nQual e a constante somada em calc_tarifa_{target:03d}? "
              "Responda so o numero.")
    return prompt, str(vals[target])


def run_task(task, port, defaults, i):
    kind = task.get("prompt_kind")
    max_tokens = task.get("max_tokens", defaults["max_tokens"])

    if kind == "longctx":
        prompt, esperado = longctx_prompt()
        r = chat(port, prompt, temp=defaults["temp"], max_tokens=max_tokens,
                 timeout=defaults["timeout"])
        ok = esperado in r["content"]
        return ok, ("" if ok else f"esperado {esperado}; veio {r['content'][:60]!r}"), r

    if kind == "tools":
        casos = [(f"Qual o clima agora em {c}?", "get_weather", "city", c)
                 for c in CITIES[:4]]
        casos += [(f"Conte quantos pedidos da tabela orders foram feitos em {m}.",
                   "run_sql", "query", "orders") for m in MONTHS[:4]]
        acertos, falhas, resps = 0, [], []
        for prompt, fn, key, needle in casos:
            r = chat(port, prompt, tools=TOOLS, temp=defaults["temp"],
                     max_tokens=600, timeout=defaults["timeout"])
            resps.append(r)
            call = r["tool_calls"][0] if r["tool_calls"] else None
            why = None
            if not call:
                why = "nao chamou tool"
            elif call["function"]["name"] != fn:
                why = f"tool errada: {call['function']['name']}"
            else:
                try:
                    args = json.loads(call["function"]["arguments"])
                    if needle.lower() not in str(args.get(key, "")).lower():
                        why = f"arg {key!r} sem {needle!r}"
                except ValueError:
                    why = "arguments nao e JSON"
            if why:
                falhas.append(f"{prompt[:30]}: {why}")
            else:
                acertos += 1
        tot = len(casos)
        agg = {
            "content": f"{acertos}/{tot}",
            "reasoning": "", "tool_calls": [],
            "finish": "tool_cases",
            "tokens": sum(x["tokens"] for x in resps),
            "ptokens": sum(x["ptokens"] for x in resps),
            "tok_s": med([x["tok_s"] for x in resps]),
            "dt": sum(x["dt"] for x in resps),
            "request": {"casos": tot},
            "raw": {"falhas": falhas},
            "score": acertos / tot,
        }
        return acertos == tot, ("" if not falhas else "; ".join(falhas[:3])), agg

    prompt = task["prompt"]
    r = chat(port, prompt, temp=defaults["temp"], max_tokens=max_tokens,
             timeout=defaults["timeout"])
    tipo = task["check"]["type"]
    if tipo == "python_tests":
        ok, why = check_python_tests(task, r)
    elif tipo == "python_files":
        ok, why = check_python_files(task, r)
    elif tipo == "json_groups":
        ok, why = check_json_groups(task, r)
    elif tipo == "contains_groups":
        ok, why = check_contains_groups(task, r)
    elif tipo == "contains_all":
        low = r["content"].lower()
        missing = [t for t in task["check"].get("terms", []) if t.lower() not in low]
        ok, why = (not missing), (f"faltou {missing}" if missing else "")
    else:
        ok, why = False, f"checker desconhecido: {tipo}"
    if not ok and r["finish"] == "length":
        why = f"estourou max_tokens; {why}"
    return ok, why, r


# ----------------------------------------------------------------- execucao
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=os.path.join(HERE, "suite.json"))
    ap.add_argument("--models", default="", help="filtra por substring (virgula)")
    ap.add_argument("--tasks", default="", help="filtra por categoria/id")
    ap.add_argument("--runs", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="nao apaga modelo baixado")
    args = ap.parse_args()

    suite = json.load(open(args.suite))
    d = {"runs": 2, "temp": 0.2, "max_tokens": 3000, "timeout": 300,
         "ctx": 32768, "port": 8099}
    d.update(suite.get("defaults") or {})
    if args.runs:
        d["runs"] = args.runs
    if args.ctx:
        d["ctx"] = args.ctx

    commit = secrets.token_hex(20)[:12]
    day = os.path.join(OUT_ROOT, time.strftime("%Y-%m-%d"))
    os.makedirs(day, exist_ok=True)
    root = os.path.join(day, commit)
    os.makedirs(root, exist_ok=True)
    print(f"suite: {os.path.basename(args.suite)} | ctx={d['ctx']} runs={d['runs']}")
    print(f"resultados: {root}")

    modelos = [m for m in suite["models"]
               if not args.models or any(
                   s.strip() in m["name"] for s in args.models.split(","))]
    tarefas = [t for t in suite["tasks"]
               if not args.tasks or any(
                   s.strip() in t["category"] or s.strip() in t["id"]
                   for s in args.tasks.split(","))]

    rows, brutos = [], []
    for m in modelos:
        print(f"\n=== {m['name']}")
        try:
            path = model_path(m)
        except Abort as e:
            print(f"   pulando: {e}")
            continue
        base = vram_used()
        proc, load_s = None, None
        try:
            proc, load_s = start_server(path, m["name"], d["ctx"], d["port"])
            _PORTS[id(proc)] = d["port"]
            print(f"   carregado em {load_s}s | VRAM {gib(vram_used())} "
                  f"(baseline {gib(base)})")
            for t in tarefas:
                res = []
                for i in range(d["runs"]):
                    try:
                        ok, why, r = run_task(t, d["port"], d, i)
                    except Abort as e:
                        print(f"   ! {t['id']}: {e}")
                        break
                    res.append((ok, r))
                    sample = {
                        "commit": commit, "modelo": m["name"], "tarefa": t["id"],
                        "categoria": t["category"], "run": i + 1, "ok": ok,
                        "motivo": why, "ctx": d["ctx"],
                        "prompt": t.get("prompt") or t.get("prompt_kind"),
                        "request": r["request"], "content": r["content"],
                        "reasoning": r["reasoning"], "tool_calls": r["tool_calls"],
                        "finish_reason": r["finish"], "tokens": r["tokens"],
                        "tok_s": r["tok_s"], "tempo_s": round(r["dt"], 2),
                        "raw": r["raw"],
                    }
                    p = os.path.join(root, f"{slug(m['name'])}-{slug(t['id'])}-run{i+1}.json")
                    json.dump(sample, open(p, "w"), ensure_ascii=False, indent=1)
                    if not ok:
                        print(f"   x {t['id']} run {i+1}: {why[:110]}")
                if not res:
                    continue
                n_ok = sum(1 for ok, _ in res if ok)
                row = {
                    "modelo": m["name"], "categoria": t["category"],
                    "tarefa": t["id"], "acertos": f"{n_ok}/{len(res)}",
                    "score": n_ok / len(res),
                    "tempo_s": med([r["dt"] for _, r in res]),
                    "tok_s": med([r["tok_s"] for _, r in res]),
                    "tokens": med([r["tokens"] for _, r in res]),
                    "ctx": d["ctx"], "load_s": load_s,
                }
                rows.append(row)
                print(f"   {t['id']}: {row['acertos']}  "
                      f"({row['tempo_s']:.1f}s, {row['tok_s'] or 0:.0f} tok/s)")
                brutos.append(row)
        except Abort as e:
            print(f"   falhou: {e}")
        finally:
            info = stop_server(proc, base)
            print(f"   unload: {'ok' if info['ok'] else 'PROBLEMA'} "
                  f"| VRAM {gib(info.get('vram'))}")
            if not args.keep and m.get("hf_file") and os.path.exists(path) \
                    and os.path.dirname(path) == DL_DIR:
                pass  # mantem o modelo: reusar entre rodadas economiza GBs

    # agregado por categoria
    cats = {}
    for r in rows:
        c = cats.setdefault(r["categoria"], {}).setdefault(r["modelo"], [])
        c.append(r)
    linhas = ["| categoria | " + " | ".join(x["name"] for x in modelos) + " |",
              "|---|" + "---|" * len(modelos)]
    for cat in cats:
        cells = []
        for m in modelos:
            rs = cats[cat].get(m["name"])
            if not rs:
                cells.append("-")
                continue
            ac = sum(float(x["score"]) for x in rs) / len(rs)
            t = med([x["tempo_s"] for x in rs])
            cells.append(f"{ac*100:.0f}% ({t:.0f}s)")
        linhas.append(f"| {cat} | " + " | ".join(cells) + " |")
    tabela = "\n".join(linhas)
    print("\n" + tabela)

    json.dump({"suite": os.path.basename(args.suite), "commit": commit,
               "ctx": d["ctx"], "runs": d["runs"], "resultados": brutos},
              open(os.path.join(root, "results.json"), "w"),
              ensure_ascii=False, indent=1)
    open(os.path.join(root, "summary.md"), "w").write(
        f"# suite {os.path.basename(args.suite)} (commit {commit}, "
        f"ctx={d['ctx']}, runs={d['runs']})\n\n{tabela}\n")
    print(f"\nsalvo em {root}")


if __name__ == "__main__":
    main()
