#!/usr/bin/env bash
# Sobe o llama-server com os parametros de local.env (usado pela unit systemd).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
set -a
# shellcheck disable=SC1091
. ./local.env
set +a
exec "$LLAMA_SERVER" \
  -m "$MODEL_PATH" \
  --jinja \
  -c "$CTX" \
  -np 1 \
  -dev "${DEVICE:-Vulkan0}" \
  -ngl "${NGL:-auto}" \
  -fit on -fitt 1024 \
  -t "$THREADS" \
  --host 127.0.0.1 --port "$PORT" \
  -lv 3
