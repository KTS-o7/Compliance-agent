#!/bin/bash
set -e

# Bootstrap .env from example if not present
if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

# Load .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

LLM_BACKEND="${LLM_BACKEND:-ollama}"

echo "=== Compliance Evaluator — Backend: $LLM_BACKEND ==="
echo ""

# ── 1. Backend-specific checks ────────────────────────────────────────────────
if [ "$LLM_BACKEND" = "ollama" ]; then
  echo "Checking Ollama is running..."
  until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    printf "."; sleep 1
  done
  echo " ready."
  echo "  Trigger:   ${OLLAMA_TRIGGER_MODEL:-ollama-evaluator/qwen3.5:9b}"
  echo "  Evaluator: ${OLLAMA_EVALUATOR_MODEL:-ollama-evaluator/qwen3.5:9b}"

elif [ "$LLM_BACKEND" = "cloud" ]; then
  echo "Using cloud backend (bifrost → Bedrock)"
  echo "  Make sure bifrost is running on localhost:24242"
fi

# ── 2. Docker services (bifrost + qdrant + app) ───────────────────────────────
echo ""
echo "Starting Docker services..."
docker compose up --build -d

echo ""
echo "Waiting for Qdrant..."
until curl -sf http://localhost:6333/readyz > /dev/null 2>&1; do
  printf "."; sleep 1
done
echo " ready."

echo "Waiting for Bifrost..."
until curl -sf http://localhost:8080/ > /dev/null 2>&1; do
  printf "."; sleep 1
done
echo " ready."

# ── 3. Cleanup on exit ────────────────────────────────────────────────────────
cleanup() {
  echo ""
  docker compose down
}
trap cleanup EXIT INT TERM

# ── 4. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "App running at:  http://localhost:8502"
echo "Bifrost UI at:   http://localhost:8080"
echo "Backend:         $LLM_BACKEND"
echo "Credentials:     senior/senior123  junior/junior123"
echo "================================================"
echo ""
docker compose logs -f app
