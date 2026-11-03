#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")" && pwd)"
# Roda dentro do container: clona o repo, instala, sobe router + mocks e testa.
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
PASS=0; FAIL=0
ok() { echo "  PASS  $1"; PASS=$((PASS+1)); }
no() { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }

echo "== dependencias"
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq git curl ca-certificates python3 >/dev/null 2>&1
echo "  git $(git --version | awk '{print $3}') | $(python3 -V)"

echo "== uv"
curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
echo "  $(uv --version)"

echo "== git clone (bundle = exatamente o commit)"
rm -rf /opt/router
git clone -q "$HERE/.router.bundle" /opt/router || { echo "clone falhou"; exit 1; }
cd /opt/router
echo "  arquivos: $(git ls-files | wc -l)"
[ -f llmrouter.py ] && ok "codigo presente no clone" || no "codigo ausente"
[ ! -f router.env ] && ok "clone NAO traz router.env" || no "router.env vazou pro clone"
[ ! -f local.env ] && ok "clone NAO traz local.env" || no "local.env vazou pro clone"
[ ! -e decisions.jsonl ] && ok "clone NAO traz decisions.jsonl" || no "decisions.jsonl vazou"
[ ! -d .venv ] && ok "clone NAO traz .venv" || no ".venv vazou"
[ ! -e memory.db ] && ok "clone NAO traz banco de memoria" || no "banco vazou"

echo "== install.sh --no-systemd"
./install.sh --no-systemd > /tmp/i1.log 2>&1; rc=$?
echo "  rc=$rc"
[ $rc -eq 0 ] && ok "install.sh roda do zero" || { no "install.sh falhou"; tail -5 /tmp/i1.log; }
grep -q "criado router.env" /tmp/i1.log && ok "install criou router.env" || no "sem router.env"
grep -q "criado local.env"  /tmp/i1.log && ok "install criou local.env"  || no "sem local.env"
./install.sh --no-systemd > /tmp/i2.log 2>&1
grep -q "router.env ja existe" /tmp/i2.log && ok "install idempotente" || no "install nao idempotente"
[ -x .venv/bin/uvicorn ] && ok "uvicorn instalado no venv" || no "uvicorn ausente"

echo "== configuracao de teste"
cat > router.env <<'ENV'
API_BASE_URL=http://127.0.0.1:9100
API_KEY=mock-key-123
API_MODEL=mock-fast
API_MODEL_THINK=mock-pro
LOCAL_BASE_URL=http://127.0.0.1:9200/v1
LOCAL_MODEL=local
LOCAL_CTX=32768
LOCAL_TOOL_ALLOWLIST=*
LOCAL_SERVICE=llama-local.service
LOCAL_AUTOSTART=0
LOCAL_IDLE_STOP_SECONDS=0
LOCAL_MIN_FREE_VRAM_MIB=0
FOOTER=1
FOOTER_REQUIRE_TOOLS=1
LAYA_MODE=off
ROUTER_DEBUG=1
ENV
chmod 600 router.env
python3 $HERE/mock_upstream.py api & MOCKAPI=$!
python3 $HERE/mock_upstream.py llama & MOCKLLAMA=$!
set -a; . ./router.env; set +a
echo "  API_BASE_URL=$API_BASE_URL LOCAL_BASE_URL=$LOCAL_BASE_URL LAYA_MODE=$LAYA_MODE"
.venv/bin/uvicorn llmrouter:app --host 127.0.0.1 --port 4000 > /tmp/router.log 2>&1 & ROUTER=$!
for i in $(seq 1 40); do curl -sf localhost:4000/v1/models >/dev/null 2>&1 && break; sleep 1; done
sleep 1

TOOLS='[{"type":"function","function":{"name":"read","description":"x","parameters":{"type":"object","properties":{}}}}]'
post() { curl -s -X POST localhost:4000/v1/chat/completions -H 'Content-Type: application/json' -d "$1"; }
hdrs() { curl -s -o /dev/null -D - -X POST localhost:4000/$1 -H 'Content-Type: application/json' -d "$2"; }

echo "== API de diagnostico"
curl -s localhost:4000/v1/models | grep -q '"router/local"' && ok "GET /v1/models" || no "GET /v1/models"
curl -s localhost:4000/local/status | grep -q '"saudavel":true' && ok "GET /local/status (llama mock no ar)" || no "local/status"
curl -s localhost:4000/local/status | grep -q '"rodape":true' && ok "rodape habilitado no status" || no "status rodape"

echo "== decisao"
R1=$(curl -s -X POST localhost:4000/route/explain -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Explique em uma frase o que e um indice B-tree."}]}')
echo "$R1" | grep -q '"destino":"local"' && ok "explain: pergunta simples -> local" || { no "explain simples"; echo "    $R1"; }
R2=$(curl -s -X POST localhost:4000/route/explain -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Refatore o arquivo parser.py e ajuste a arquitetura do projeto."}]}')
echo "$R2" | grep -q '"destino":"api"' && ok "explain: refatorar/arquitetura -> api" || { no "explain pesado"; echo "    $R2"; }

echo "== non-stream"
B1=$(post "{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"Diga oi\"}],\"tools\":$TOOLS}")
echo "$B1" | grep -q "resposta local" && ok "local: conteudo" || { no "local conteudo"; echo "    ${B1:0:200}"; }
echo "$B1" | grep -q "ROTA: LOCAL (llama.cpp)" && ok "local: rodape LOCAL" || { no "rodape local"; echo "    ${B1: -200}"; }
echo "$B1" | grep -q "tok/s" && ok "local: tok/s do timings" || no "tok/s"
B2=$(post "{\"model\":\"api\",\"messages\":[{\"role\":\"user\",\"content\":\"Diga oi\"}],\"tools\":$TOOLS}")
echo "$B2" | grep -q "resposta da api" && ok "api: conteudo" || no "api conteudo"
echo "$B2" | grep -q "ROTA: API (mock-api)" && ok "api: rodape API" || no "rodape api"

echo "== gzip do upstream"
G=$(curl -s -H 'Accept-Encoding: gzip' -X POST localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"api\",\"messages\":[{\"role\":\"user\",\"content\":\"oi\"}],\"tools\":$TOOLS}")
case "$G" in '{"id"'*) ok "gzip decodificado (JSON legivel no cliente)";; *) no "gzip quebrado: ${G:0:40}";; esac

echo "== stream"
curl -sN -X POST localhost:4000/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"Diga oi\"}],\"stream\":true,\"tools\":$TOOLS}" > /tmp/sse.txt
if python3 $HERE/check_stream.py /tmp/sse.txt; then ok "ordem do SSE (conteudo->rodape->finish->DONE)"; else no "ordem do SSE"; fi
grep -c "ROTA: LOCAL" /tmp/sse.txt >/dev/null 2>&1 && ok "rodape presente no stream" || no "sem rodape no stream"

echo "== escopo do rodape e roteamento"
NT=$(post '{"model":"auto","messages":[{"role":"user","content":"Diga oi"}],"max_tokens":10}')
echo "$NT" | grep -q "ROTA" && no "rodape apareceu em request sem tools" || ok "request sem tools: sem rodape"
hdrs v1/chat/completions '{"model":"auto","messages":[{"role":"user","content":"oi"},{"role":"tool","content":"resultado"}]}' | grep -qi "x-routed-to: api" && ok "loop de tools -> api" || no "loop de tools"
FB=$(curl -s -D /tmp/fh -X POST localhost:4000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"BOOM"}],"tools":[{"type":"function","function":{"name":"read","description":"x","parameters":{"type":"object","properties":{}}}}]}')
grep -qi "x-routed-to: api" /tmp/fh && ok "local com 500 -> fallback para api" || no "fallback"
echo "$FB" | grep -q "resposta da api" && ok "fallback entregou resposta da api" || no "fallback sem resposta"
grep -qi "x-route-reason: fallback" /tmp/fh && ok "motivo do fallback no header" || no "motivo do fallback"

echo "== log do router"
grep -E "\[router\]" /tmp/router.log | tail -12

echo "== aviso de mudanca de rota"
post "{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"oi\"}],\"tools\":$TOOLS}" > /dev/null
CH=$(post "{\"model\":\"api\",\"messages\":[{\"role\":\"user\",\"content\":\"oi\"}],\"tools\":$TOOLS}")
echo "$CH" | grep -q "mudou: LOCAL -> API" && ok "rodape avisa mudanca de rota" || { no "sem aviso de mudanca de rota"; echo "    ${CH: -240}"; }

echo "== pro no pensamento, flash na execucao"
P=$(post "{\"model\":\"api\",\"messages\":[{\"role\":\"user\",\"content\":\"Planeje a arquitetura de um servico de pagamentos e explique os trade-offs.\"}],\"tools\":$TOOLS}")
echo "$P" | grep -q "ROTA: API (mock-pro)" && ok "planejamento/arquitetura -> pro" || { no "planejamento nao foi para o pro"; echo "    ${P: -170}"; }
X=$(post "{\"model\":\"api\",\"messages\":[{\"role\":\"user\",\"content\":\"Diga oi\"}],\"tools\":$TOOLS}")
echo "$X" | grep -q "ROTA: API (mock-fast)" && ok "execucao simples -> flash" || { no "execucao nao foi para o flash"; echo "    ${X: -170}"; }
L=$(post "{\"model\":\"api\",\"messages\":[{\"role\":\"user\",\"content\":\"Planeje a arquitetura\"},{\"role\":\"tool\",\"content\":\"ok\"}],\"tools\":$TOOLS}")
echo "$L" | grep -q "ROTA: API (mock-fast)" && ok "loop de tools -> flash" || { no "loop nao caiu no flash"; echo "    ${L: -170}"; }

echo "== sem systemd e sem modelo local (fast-fail)"
kill $MOCKLLAMA 2>/dev/null; sleep 1
T0=$(date +%s%N)
SF=$(curl -s -X POST localhost:4000/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"auto\",\"messages\":[{\"role\":\"user\",\"content\":\"Diga oi\"}],\"tools\":$TOOLS}")
T1=$(date +%s%N); MS=$(( (T1-T0)/1000000 ))
echo "$SF" | grep -q "resposta da api" && ok "local fora + sem systemd -> api" || no "nao caiu pra api"
[ "$MS" -lt 5000 ] && ok "fallback rapido (${MS}ms, sem esperar 240s)" || no "fallback lento (${MS}ms)"

echo "== install.sh sem systemd (degradacao)"
./install.sh > /tmp/i3.log 2>&1; rc=$?
[ $rc -eq 0 ] && ok "install.sh nao quebra sem systemctl" || { no "install.sh falhou sem systemctl"; tail -3 /tmp/i3.log; }
grep -q "systemctl --user indisponivel" /tmp/i3.log && ok "install avisa e segue" || no "install nao avisou"

kill $ROUTER $MOCKAPI 2>/dev/null
echo
echo "======================================"
echo "RESULTADO: $PASS pass, $FAIL fail"
echo "======================================"
[ $FAIL -eq 0 ]
