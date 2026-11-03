#!/usr/bin/env bash
# Instala o llmrouter nesta maquina (Linux, systemd --user).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

FORCE=0
WITH_LAYA=0
NO_SYSTEMD=0

usage() {
  cat <<'EOF'
Uso: ./install.sh [--with-laya] [--force] [--no-systemd]

  --with-laya   instala torch (CPU) + laya, o decisor local opcional (+1.7 GB)
  --force       sobrescreve units existentes em ~/.config/systemd/user
  --no-systemd  so prepara venv e arquivos de config
EOF
}

for arg in "$@"; do
  case "$arg" in
    --with-laya) WITH_LAYA=1 ;;
    --force) FORCE=1 ;;
    --no-systemd) NO_SYSTEMD=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "argumento desconhecido: $arg"; usage; exit 1 ;;
  esac
done

command -v uv >/dev/null || {
  echo "uv nao encontrado. Instale:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh"
  exit 1
}

echo "== venv"
if [ ! -d .venv ]; then
  uv venv .venv
fi
uv pip install -r requirements.txt

if [ "$WITH_LAYA" = 1 ]; then
  echo "== laya (+ torch cpu)"
  uv pip install torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install -r requirements-laya.txt
fi

echo "== arquivos de configuracao"
if [ ! -f router.env ]; then
  cp router.env.example router.env
  chmod 600 router.env
  echo "  criado router.env  -> edite API_KEY/API_BASE_URL/API_MODEL"
else
  echo "  router.env ja existe (mantido)"
fi
if [ ! -f local.env ]; then
  cp local.env.example local.env
  echo "  criado local.env   -> edite LLAMA_SERVER/MODEL_PATH"
else
  echo "  local.env ja existe (mantido)"
fi

if [ "$NO_SYSTEMD" = 1 ]; then
  echo
  echo "Pronto (sem systemd). Rode o router com:"
  echo "  set -a; . ./router.env; set +a; .venv/bin/uvicorn llmrouter:app --host 127.0.0.1 --port 4000"
  exit 0
fi

if ! command -v systemctl >/dev/null 2>&1 || ! systemctl --user is-system-running >/dev/null 2>&1; then
  echo "  systemctl --user indisponivel; pulando units (rode o router na mao)"
  exit 0
fi

echo "== systemd --user"
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"
for u in llama-local.service llmrouter.service; do
  src="systemd/$u"
  dst="$UNIT_DIR/$u"
  rendered="$(sed "s|@REPO@|$REPO|g" "$src")"
  if [ -f "$dst" ]; then
    if [ "$(cat "$dst")" = "$rendered" ]; then
      echo "  $u ja esta igual, nada a fazer"
      continue
    fi
    if [ "$FORCE" = 0 ]; then
      echo "  AVISO: $dst difere do template; mantido (use --force para sobrescrever)"
      continue
    fi
    cp "$dst" "$dst.bak.$(date +%Y%m%d-%H%M%S)"
  fi
  printf '%s\n' "$rendered" > "$dst"
  echo "  instalado $u"
done

systemctl --user daemon-reload
systemctl --user enable llmrouter.service >/dev/null
systemctl --user disable llama-local.service >/dev/null 2>&1 || true
systemctl --user restart llmrouter.service

echo
echo "== proximos passos"
echo "1. edite $REPO/router.env (API_KEY, API_BASE_URL, API_MODEL)"
echo "2. edite $REPO/local.env  (LLAMA_SERVER e MODEL_PATH do .gguf)"
echo "   modelo sugerido: unsloth/gpt-oss-20b-GGUF (UD-Q4_K_XL)"
echo "3. systemctl --user restart llmrouter"
echo "4. curl -s http://127.0.0.1:4000/local/status"
echo
echo "O llama-local sobe sob demanda e cai depois de LOCAL_IDLE_STOP_SECONDS."
