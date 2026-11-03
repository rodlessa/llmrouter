#!/usr/bin/env bash
# Roda a suite em container Debian limpo, a partir do commit atual.
# Uso: tests/run.sh   (usa podman se existir, senao docker; repo precisa estar commitado)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

if [ -n "$(git -C "$REPO" status --porcelain)" ]; then
  echo "ERRO: ha mudancas nao commitadas. O teste clona o ultimo COMMIT," >&2
  echo "      entao ele nao veria o que esta no working tree." >&2
  echo "      Commite antes (ou rode com --dirty para testar o commit atual)." >&2
  [ "${1:-}" = "--dirty" ] || exit 1
fi

git -C "$REPO" bundle create "$HERE/.router.bundle" --all >/dev/null
trap 'rm -f "$HERE/.router.bundle"' EXIT

if command -v podman >/dev/null 2>&1; then
  ENGINE=podman
elif command -v docker >/dev/null 2>&1; then
  ENGINE=docker
else
  echo "precisa de podman ou docker" >&2
  exit 1
fi

echo "engine: $ENGINE"
exec "$ENGINE" run --rm -v "$HERE:/tests:ro" debian:bookworm-slim bash /tests/in-container.sh
