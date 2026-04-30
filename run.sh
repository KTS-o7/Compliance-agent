#!/bin/bash
set -e
echo "=== Compliance Evaluator ==="
echo "Starting services..."
docker compose up --build -d
echo ""
echo "Waiting for Qdrant..."
until curl -sf http://localhost:6333/readyz > /dev/null 2>&1; do
  printf "."; sleep 1
done
echo " ready."
echo ""
echo "App running at: http://localhost:8502"
echo "Credentials: senior/senior123 or junior/junior123"
docker compose logs -f app
