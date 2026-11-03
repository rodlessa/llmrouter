"""Valida a ordem do SSE entregue pelo router: conteudo -> rodape -> finish -> DONE."""
import json
import sys

raw = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
events, done_at, footer_at, finish_at = [], None, None, None
text = ""
for i, line in enumerate(raw):
    if not line.startswith("data:"):
        continue
    payload = line[5:].strip()
    if payload == "[DONE]":
        done_at = i
        continue
    try:
        obj = json.loads(payload)
    except ValueError:
        print(f"FAIL: chunk nao-JSON na linha {i}: {line[:80]}")
        sys.exit(1)
    events.append(obj)
    for ch in obj.get("choices") or []:
        d = (ch.get("delta") or {}).get("content") or ""
        if d:
            text += d
        if "ROTA:" in d:
            footer_at = i
        if ch.get("finish_reason"):
            finish_at = i

ok = True
if done_at is None:
    print("FAIL: sem [DONE]"); ok = False
if footer_at is None:
    print("FAIL: rodape nao apareceu no conteudo"); ok = False
if finish_at is None:
    print("FAIL: sem finish_reason"); ok = False
if footer_at and finish_at and not (footer_at < finish_at):
    print(f"FAIL: rodape (linha {footer_at}) depois do finish (linha {finish_at})"); ok = False
if done_at and finish_at and not (finish_at < done_at):
    print("FAIL: finish depois do [DONE]"); ok = False
if done_at is not None and done_at != max(
        i for i, l in enumerate(raw) if l.startswith("data:")):
    print("FAIL: evento depois do [DONE]"); ok = False
if "resposta local" not in text.replace("\n", " "):
    print(f"FAIL: texto inesperado: {text[:120]!r}"); ok = False
print("PASS: SSE ok, ordem conteudo->rodape->finish->DONE" if ok else "FAIL: SSE")
sys.exit(0 if ok else 1)
