# Startup Guide

Step-by-step instructions to get the Compliance Evaluator running from a fresh machine.

---

## What Gets Started

| Service | How | Port |
|---------|-----|------|
| Qdrant (vector store) | Docker container | 6333 |
| Bifrost (LLM proxy) | Docker container | 8080 |
| Streamlit app | Docker container | 8502 |
| Ollama (LLM runtime) | Host process | 11434 |

Rules are seeded into Qdrant automatically on first app startup. The BGE embedding model (`BAAI/bge-base-en-v1.5`) is baked into the Docker image — no separate download needed.

---

## Prerequisites

### 1. Docker Desktop

Download and install from https://www.docker.com/products/docker-desktop

Make sure Docker Desktop is **open and running** before proceeding. You can verify:

```bash
docker info
```

Expected: prints engine info without errors.

---

### 2. Ollama

**macOS:**
```bash
brew install ollama
```

Or download the installer from https://ollama.com and follow the prompts.

Verify:
```bash
ollama --version
```

---

### 3. System Requirements

| Requirement | Minimum |
|-------------|---------|
| RAM | 16 GB (18 GB unified recommended) |
| Disk | 10 GB free (5 GB model + 4 GB Docker images) |
| OS | macOS 13+ (Apple Silicon recommended) or Linux |
| Docker Desktop | 4.x or later |
| Ollama | 0.19 or later |

> **Apple Silicon note:** Ollama 0.19+ uses MLX natively on M-series chips. This gives the best local inference performance. Intel Mac and Linux will use CPU inference and will be significantly slower.

---

## First-Time Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/KTS-o7/Compliance-agent.git
cd Compliance-agent
```

---

### Step 2 — Pull the LLM model

```bash
ollama pull qwen3.5:9b
```

This downloads ~5 GB. Only needed once. Progress is shown in the terminal.

To verify the model is available:
```bash
ollama list
```

Expected output includes `qwen3.5:9b`.

---

### Step 3 — Start Ollama

Open a dedicated terminal and run:

```bash
ollama serve
```

Keep this terminal open for the duration of the session. Ollama must be running before starting the app.

---

### Step 4 — Run the app

Open a second terminal in the project directory:

```bash
./run.sh
```

`run.sh` will:
1. Create `.env` from `.env.example` automatically (first run only)
2. Confirm Ollama is reachable at `localhost:11434`
3. Build and start Docker containers (Qdrant + Bifrost + app)
4. Wait for Qdrant and Bifrost to be healthy
5. Print the app URL and credentials
6. Tail app logs

First run takes ~2–3 minutes to build the Docker image (BGE model is downloaded into the image during build).

---

### Step 5 — Open the app

```
http://localhost:8502
```

Login credentials:

| Username | Password | Access |
|----------|----------|--------|
| `senior` | `senior123` | Reviewer + Admin + Audit Log |
| `junior` | `junior123` | Reviewer only |

---

## What Happens Automatically

| Action | When | Notes |
|--------|------|-------|
| `.env` created | First `./run.sh` | Copied from `.env.example` |
| Docker images pulled | First `./run.sh` | Qdrant, Bifrost, Python base |
| BGE model baked in | Docker image build | `BAAI/bge-base-en-v1.5`, 768-dim |
| Qdrant collection created | App startup | `compliance_rules` collection |
| 10 seed rules inserted | App startup | FDCPA, CFPB, FCRA, Bankruptcy statutes |
| BGE model loaded into memory | After login | Prefetched before first evaluation |

---

## Stopping the App

Press `Ctrl+C` in the `run.sh` terminal. It automatically runs `docker compose down`.

To also remove persisted Qdrant data and audit log:
```bash
docker compose down -v
```

> **Note:** Running `docker compose down -v` deletes the Qdrant vector store. Rules will be re-seeded automatically on the next startup.

---

## Subsequent Runs

After the first-time setup, every subsequent run is just:

```bash
# Terminal 1
ollama serve

# Terminal 2
./run.sh
```

No model re-download, no config changes, no manual steps.

---

## Troubleshooting

### `run.sh` hangs at "Checking Ollama is running..."

Ollama is not running. In a separate terminal:
```bash
ollama serve
```

### Docker build fails with network errors

Docker Desktop is not running. Open it from your Applications folder and wait for the whale icon to appear in the menu bar.

### App loads but evaluations time out

The `qwen3.5:9b` model was not pulled. Run:
```bash
ollama pull qwen3.5:9b
```

### Port 8502 already in use

Another process is using the port. Find and stop it:
```bash
lsof -i :8502
kill -9 <PID>
```

### Qdrant rules missing after restart

If you ran `docker compose down -v`, the volume was deleted. Rules are re-seeded automatically — just log in and they will be there.

### `ollama serve` fails with "address already in use"

Ollama is already running as a background process. This is fine — `run.sh` will detect it and proceed.

---

## Running the QoS Eval (Optional)

To reproduce the measured accuracy and latency numbers:

```bash
# Requires Python 3.11 and dependencies installed locally
pip install -r requirements.txt
python3.11 eval/run_eval.py
```

> The app (Docker) and Ollama must both be running. Results are saved to `eval/qos_results.json`.

---

## Project Ports Summary

| Port | Service | Notes |
|------|---------|-------|
| 8502 | Streamlit app | Main UI |
| 8080 | Bifrost | LLM proxy dashboard |
| 6333 | Qdrant HTTP | Vector store API |
| 6334 | Qdrant gRPC | Vector store gRPC |
| 11434 | Ollama | LLM runtime (host, not Docker) |
