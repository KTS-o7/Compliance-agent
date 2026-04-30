#!/bin/bash
set -e

echo "=== Compliance Evaluator — Phase 2 (MLX + Bifrost) ==="
echo ""

# ── 1. MLX model servers ──────────────────────────────────────────────────────
echo "Starting MLX model servers..."

# Kill any existing instances on these ports
lsof -ti:8081 | xargs kill -9 2>/dev/null || true
lsof -ti:8082 | xargs kill -9 2>/dev/null || true

python3.11 -m mlx_lm.server \
  --model mlx-community/Qwen3-4B-4bit \
  --port 8081 --host 0.0.0.0 \
  --chat-template-args '{"enable_thinking":false}' \
  > /tmp/mlx-trigger.log 2>&1 &
MLX_TRIGGER_PID=$!

python3.11 -m mlx_lm.server \
  --model mlx-community/Qwen3.5-9B-MLX-4bit \
  --port 8082 --host 0.0.0.0 \
  --chat-template-args '{"enable_thinking":false}' \
  > /tmp/mlx-evaluator.log 2>&1 &
MLX_EVALUATOR_PID=$!

echo "  Trigger model  (Qwen3-4B)    → pid $MLX_TRIGGER_PID  port 8081"
echo "  Evaluator model (Qwen3.5-9B) → pid $MLX_EVALUATOR_PID port 8082"

# Wait for both MLX servers to be ready
echo ""
echo "Waiting for MLX trigger server (port 8081)..."
until curl -sf http://localhost:8081/v1/models > /dev/null 2>&1; do
  printf "."; sleep 2
done
echo " ready."

echo "Waiting for MLX evaluator server (port 8082)..."
until curl -sf http://localhost:8082/v1/models > /dev/null 2>&1; do
  printf "."; sleep 2
done
echo " ready."

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
  echo "Shutting down MLX servers..."
  kill $MLX_TRIGGER_PID $MLX_EVALUATOR_PID 2>/dev/null || true
  docker compose down
}
trap cleanup EXIT INT TERM

# ── 4. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "App running at:  http://localhost:8502"
echo "Bifrost UI at:   http://localhost:8080"
echo "Credentials:     senior/senior123  junior/junior123"
echo "MLX trigger log: /tmp/mlx-trigger.log"
echo "MLX eval log:    /tmp/mlx-evaluator.log"
echo "================================================"
echo ""
docker compose logs -f app
