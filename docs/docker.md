# Docker Deployment

This guide walks you through deploying AgenDino using Docker Compose.

---

## Requirements

- **Docker** and **Docker Compose** or **Docker Desktop** installed on your system.
- A **Google Gemini API key** for transcription, summarization, RAG, and daily recaps.
- _(Optional)_ A **Notion API key** and parent page ID for publishing summaries.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DStt/agendino.git
cd agendino
```

## Configuration

Before running, update the `compose.yaml` file with your specific values:

- Replace `{local-dir-here}` with absolute or relative paths to local directories on your host machine:
  - For recordings: e.g., `/path/to/your/recordings`
  - For settings: e.g., `/path/to/your/settings`
  - For Traefik certificates: e.g., `/path/to/your/certs`

- Replace `{internal-ip-here}` with your server's internal IP address or domain name (e.g., `192.168.1.100` or `agendino.local`).

Create a `.env` file in the project root (same as in local setup):

```env
# Required - Google Gemini API key
GEMINI_API_KEY=your-gemini-api-key

# Optional - Gemini model names (defaults shown)
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=text-embedding-001

# Optional - Notion integration
NOTION_API_KEY=your-notion-integration-token
NOTION_PAGE_ID=your-notion-parent-page-id

# Optional - SQLite database name (default: agendino.db)
DATABASE_NAME=agendino.db

# Optional - Enable login authentication (default: false)
AUTH_ENABLED=false

# Optional - Local Whisper transcription settings
WHISPER_MODEL_SIZE=small          # tiny | base | small | medium | large-v3
WHISPER_DEVICE=cpu                # cpu | cuda
WHISPER_COMPUTE_TYPE=auto         # auto | int8 | float16 | float32
```

See [Authentication](authentication.md) for details on `AUTH_ENABLED` and [Transcription](transcription.md) for Whisper settings.

## Running the Deployment

```bash
docker compose up -d
```

This starts the services in detached mode. AgenDino will be accessible via Traefik at the configured host (e.g., `https://{internal-ip-here}`), and the Traefik dashboard at `https://{internal-ip-here}/traefik/dashboard/`.

To view logs:

```bash
docker compose logs -f
```

To stop:

```bash
docker compose down
```

## GPU acceleration (optional)

Local Whisper transcription runs on the **celery** worker. By default it runs on **CPU**.
You can optionally run it on an **NVIDIA GPU** by layering in `compose.gpu.yaml`.

> GPU support targets **Linux** hosts. The container image is Linux-based regardless of where
> Docker runs.

### Host prerequisites

1. NVIDIA driver installed — `nvidia-smi` works on the host.
2. Install the **NVIDIA Container Toolkit** and configure the Docker runtime:
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
3. Verify Docker can see the GPU:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

### Enable GPU

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
```

Or make it the default so you don't have to pass the flags — add this line to `.env`:

```env
COMPOSE_FILE=compose.yaml:compose.gpu.yaml
```

Then plain `docker compose up -d` uses the GPU.

### Disable GPU

Run plain `docker compose up -d` (CPU mode), or comment out the `COMPOSE_FILE` line in `.env`.

**Notes on `COMPOSE_FILE`:**
- A command-line `-f` overrides `COMPOSE_FILE`, so you can still force CPU with
  `docker compose -f compose.yaml up -d` even when the `.env` default is GPU.
- It is "sticky": once set, **every** `docker compose` command uses both files. On a host
  without a working GPU, plain `docker compose up` then fails on the GPU reservation — use the
  base file only on non-GPU hosts.
- The path separator is `:` on Linux (`;` on Windows).

`compose.gpu.yaml` sets `WHISPER_COMPUTE_TYPE=float16` and `WHISPER_MODEL_SIZE=large-v3`,
which suit a modern GPU. For older cards, adjust these — see
[Transcription → GPU compatibility](transcription.md).

## Concurrency tuning (CPU vs GPU)

`CELERY_CONCURRENCY` controls how many transcription/summarization jobs run in **parallel**
on the worker. Set it in `.env` or the `celery` service `environment:` (default `1`). Jobs for
the *same* recording are deduplicated by a lock, so this only speeds up processing of
*different* recordings at once.

> Uploads scale separately via `WEB_CONCURRENCY` (web workers) — a different knob with a
> different bottleneck.

**To change it safely:** raise `CELERY_CONCURRENCY` by 1, recreate the worker
(`docker compose up -d` — no rebuild needed for an env change), start 2+ transcriptions on
different files, and watch the limiting resource before going higher.

| Mode | Limited by | Recommended | Why |
|------|-----------|-------------|-----|
| **CPU** | Cores / RAM | **1–2** | faster-whisper already multithreads internally; stacking workers oversubscribes the CPU and can be *slower*. Each worker also loads its own model into RAM. |
| **GPU** | VRAM | **1–2** (start at 1) | Each worker loads its own model copy into VRAM and they share one GPU's compute. Estimate `floor(free_VRAM / model_VRAM)`. On a 10 GB card with `large-v3` float16 (~3 GB), 1–2 is safe; an OOM crashes the task. |

---

**Next:** explore the features - start with [Recording Management](recording-management.md) or browse the full [Documentation Index](index.md).
