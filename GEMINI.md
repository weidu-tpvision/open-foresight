# 🔭 GEMINI.md: AI Agent System Architecture & Engineering Context

> **Target Audience:** Antigravity / Gemini / Autonomous Coding Agents  
> **Repository:** OpenForesight (Spark-Foresight)  
> **Role:** Canonical engineering context, architectural blueprint, operational manual, and invariant guide for AI coding agents developing, debugging, and extending this codebase.

---

## 🎯 1. Mission & Non-Negotiable System Invariants

### 1.1 The Product Mission
OpenForesight provides a 100% on-device open-source implementation of **Google AI Edge Foresight** for **Linux systems with NVIDIA GPUs**. It eliminates centralized cloud SaaS meeting tools (e.g., Otter.ai, Granola, Fireflies) by executing speech recognition, temporal note synchronization, document retrieval, and generative synthesis entirely locally.

### 1.2 The Core Invariants (Never Break These)
When writing, modifying, or testing code in this repository, you **MUST** strictly adhere to these invariants:

1. 🔒 **Zero Cloud Egress & Zero Data Upload:**
   - All audio streams, voice recordings, transcripts, notes, vector embeddings, and LLM inferences **MUST** remain strictly resident on `localhost`.
   - Under no circumstances may code or scripts transmit audio, speech segments, transcripts, or document text to any external cloud API or remote server.
   - The LLM and embedding pipelines must operate entirely against local runtimes (e.g., local Ollama on `http://localhost:11434`, TensorRT-LLM, vLLM, or llama.cpp).

2. 🛡️ **Absolute Privacy Boundary: Never Commit or Upload Private Data:**
   - **No Databases or Transcripts:** Meeting session databases (`foresight.db`, `*.db`, `*.sqlite`, `*.db-wal`, `*.db-shm`), transcript logs, and vector stores contain confidential user discussions and indexed knowledge. They **MUST NEVER** be committed to Git, uploaded, or exposed.
   - **No Audio Recordings or Processing Dumps:** Voice recordings, raw PCM buffers, `.wav` files, loopback audio captures, and intermediate VAD speech chunks contain sensitive voice data and private conversations. They **MUST NEVER** be staged, committed, or pushed to any repository.
   - **No Personal Identifiers:** Do **NOT** commit hardcoded user home directories (`/home/<user>/...`), personal usernames, corporate entities, proprietary project paths, or private hostnames. Always use dynamic path resolution (e.g., `Path.cwd()`, relative paths, or environment variables).
   - **Internal Documentation Boundary:** Internal planning specifications, architecture explorations, task breakdowns, code reviews, retrospective notes, and learning logs (e.g., `*PLAN*.md`, `*REPORT*.md`, `*REVIEW*.md`, `*LEARNING*.md`, `planning/`, `reviews/`) are strictly for local pair-programming. They **MUST NOT** be committed or pushed to public Git repositories.
   - Maintain strict `.gitignore` rules to automatically enforce these exclusions at all times.

3. ⏱️ **Strict Temporal Provenance & Zero Hallucination:**
   - The shorthand enrichment engine must **never** invent or hallucinate action items.
   - Every synthesized action item must cite an exact conversational time reference `[ref: MM:SS]` derived from a correlated transcript window ($[T_{\text{note}} - 45\text{s}, \; T_{\text{note}} + 15\text{s}]$) or cite an exact indexed document section.

4. 🧪 **Test Suite Integrity:**
   - Every commit must preserve the passing state of all tests in `tests/` (`PYTHONPATH=. .venv/bin/pytest tests/`).

---

## 🖥️ 2. Execution Environment & Hardware Architecture

The system is designed for modern Linux workstations, servers, and developer platforms:

| Layer | Specification | Details & Handling in Code |
| :--- | :--- | :--- |
| **OS** | Linux (Kernel 6.x / 7.x) | Verified on Ubuntu 24.04 LTS; compatible with Fedora, Debian, Arch. |
| **CPU Architecture** | `x86_64` or `aarch64` | Pure cross-platform Python 3.12+; no architecture-specific assembly. |
| **GPU Compute** | NVIDIA GPU with CUDA | PyTorch 2.x with CUDA 12/13. Tested on Grace Blackwell GB10 and desktop RTX GPUs. |
| **Audio Server** | PipeWire + WirePlumber | Uses `pw-cli` for dynamic node discovery and `pw-record` for raw PCM capture. |
| **Local LLM Models** | Ollama (`localhost:11434`) | **`gemma4:e4b`** for generative note enrichment; **`embeddinggemma-2`** for 768-D vectors. |
| **Persistence** | SQLite in WAL mode | Single local file `foresight.db` with PRAGMA synchronous = NORMAL. |

---

## 🏛️ 3. Subsystem Deep Dive & Codebase Layout

```text
open-foresight/
│
├── daemon/                          # Core Python Backend Daemon
│   ├── app.py                       # FastAPI entrypoint, REST APIs, and WebSocket routes
│   ├── session_manager.py           # Live session state machine, audio tap coordinator & broadcaster
│   │
│   ├── audio/                       # Hardware Audio Plumbing
│   │   ├── capture.py               # PipeWire dual-stream capture (Mic Source + Output Sink Monitor)
│   │   └── vad.py                   # Silero VAD streaming speech chunker (16kHz mono PCM)
│   │
│   ├── asr/                         # Speech Recognition on CUDA
│   │   ├── engine.py                # Whisper ASR engine (Whisper on CUDA with word timestamps)
│   │   └── timestamps.py            # Millisecond-to-string formatters and alignment utilities
│   │
│   ├── sync/                        # The Shorthand Synchronization Engine (Core Magic)
│   │   └── enricher.py              # Window slicing [T-45s, T+15s] + Gemma 4 executive minute generator
│   │
│   ├── rag/                         # Local Multimodal Knowledge Base & Assistant
│   │   ├── chunker.py               # Markdown (heading-aware) and PDF (pypdf page tracking) chunker
│   │   ├── embeddings.py            # EmbeddingGemma 2 client (768-D dense vectors via Ollama)
│   │   ├── vector_store.py          # SQLite vector store with cosine similarity search
│   │   ├── watcher.py               # Autonomous background watchdog folder observer for docs/
│   │   └── assistant.py             # Multi-source Live Assistant (combines docs + meeting transcripts)
│   │
│   ├── llm/                         # Local LLM Client
│   │   └── client.py                # Async HTTP client for Ollama /api/generate (gemma4:e4b)
│   │
│   ├── db/                          # Database & Persistence Layer
│   │   ├── schema.sql               # SQLite DDL schema (sessions, transcripts, notes, docs, chunks)
│   │   └── repository.py            # Async SQLite data access repository (aiosqlite)
│   │
│   └── static/                      # Ambient Web Studio UI
│       └── index.html               # Single-page meeting intelligence studio (Tailwind + WebSockets)
│
├── docs/                            # Watched knowledge folder (PDFs, Markdown)
│   └── architecture_overview.md     # Default verified specification document
│
├── tests/                           # Complete Verification Suite (15 passing tests)
│   ├── test_api_endpoints.py        # REST API endpoints, audio device queries, session lifecycle
│   ├── test_asr_transcription.py    # GPU Whisper ASR pipeline, Silero VAD, SQLite commits
│   ├── test_db.py                   # SQLite schema, WAL mode, session tracking, temporal queries
│   ├── test_shorthand_engine.py     # Gemma 4 LLM client, EmbeddingGemma 2, temporal note synthesis
│   └── test_rag_pipeline.py         # Document chunker, vector store, and Live Assistant grounding
│
├── index_workspace.py               # CLI tool to batch index any directory with Rich progress
├── run_live_demo.py                 # Interactive terminal live recording & transcription demo
├── pyproject.toml                   # Project configuration and uv dependencies
└── README.md                        # Public-facing documentation
```

---

## 🔍 4. Key Subsystem Workflows for Agents

### 4.1 Subsystem 1: Dual Audio Capture & VAD (`daemon/audio/`)
- `discover_audio_nodes()` in `capture.py`: Spawns `pw-cli list-objects Node` and parses output into `AudioNode` objects (`Audio/Source` for mics, `Audio/Sink` for monitors).
- `PipeWireAudioStream`: Spawns asynchronous `pw-record --target <node> --rate 16000 --channels 1 --format s16 -` and reads raw 16kHz mono PCM chunks.
- `SileroVADSegmenter` in `vad.py`: Feeds 512-sample (32ms) frames to Silero VAD. Applies silence padding (300ms) and minimum speech duration (250ms) to emit clean speech bursts for ASR.

### 4.2 Subsystem 2: Streaming ASR on CUDA (`daemon/asr/`)
- `ASREngine` in `engine.py`: Loads Whisper (`openai/whisper-tiny` by default or larger models) on GPU (`cuda:0`).
- Transcribes raw PCM float32 arrays into text chunks with millisecond start/end timestamps.
- Commits each segment immediately to SQLite `transcript_segments` via `DatabaseRepository.add_transcript_segment()`.
- Emits real-time WebSocket payloads to connected frontend clients.

### 4.3 Subsystem 3: The Shorthand Engine (`daemon/sync/`)
- Frontend logs monotonic offset $T_{\text{note}}$ (ms since session start) on every keystroke.
- When the user requests enrichment, `ShorthandEnricher.enrich_shorthand()`:
  1. Computes conversational window: $\mathcal{W} = [T_{\text{note}} - 45000\text{ms}, \; T_{\text{note}} + 15000\text{ms}]$.
  2. Queries SQLite for all transcript segments overlapping $\mathcal{W}$.
  3. Prompts **`gemma4:e4b`** with strict system instructions to expand cryptic shorthand bullets into structured executive minutes citing `[ref: MM:SS]`.

### 4.4 Subsystem 4: Local Knowledge Base & Vector RAG (`daemon/rag/`)
- `DocumentChunker` in `chunker.py`: Splits Markdown by headers and sliding window (1,200 chars with 120 char overlap); extracts PDF text with page numbers via `pypdf`.
- `EmbeddingGemmaClient` in `embeddings.py`: Batches text chunks in slices of 32 to Ollama `/api/embed` with model `embeddinggemma-2` using persistent HTTP connection pooling to produce exact 768-D dense vectors.
- `VectorStore` in `vector_store.py`: Stores chunks in SQLite `document_chunks` with embeddings serialized as binary float32 blobs (`np.array().tobytes()`). Performs BLAS-vectorized matrix-vector cosine similarity across stacked arrays in SQLite memory with incremental file modification checks.
- `LiveAssistant` in `assistant.py`: Dual-grounding RAG that searches both local document chunks and live meeting transcript segments, prompting Gemma 4 to synthesize grounded answers with citation badges.

---

## ⚡ 5. Developer & Agent CLI Commands

Always run these commands from the repository root with the virtual environment:

### Run the Full Verification Suite
```bash
PYTHONPATH=. .venv/bin/pytest tests/
```
*Expected Result:* 15 passed, 0 failed.

### Launch the Daemon & Web Studio
```bash
.venv/bin/uvicorn daemon.app:app --host 0.0.0.0 --port 8000
```
*Web UI:* [http://localhost:8000](http://localhost:8000)  
*Swagger API Docs:* [http://localhost:8000/docs](http://localhost:8000/docs)

### Run Interactive Live Terminal Capture Demo
```bash
.venv/bin/python run_live_demo.py
```

### Batch Index a Local Workspace or Folder
```bash
# Index current directory:
.venv/bin/python index_workspace.py .

# Index a specific project folder:
.venv/bin/python index_workspace.py /path/to/project
```

### Test Backend Endpoints via curl
```bash
# Check telemetry & GPU availability
curl http://localhost:8000/api/health

# List detected audio nodes
curl http://localhost:8000/api/audio/devices

# Check configured Ollama models
curl http://localhost:8000/api/models

# List indexed documents
curl http://localhost:8000/api/knowledge/documents

# Ask Live Assistant a question
curl -X POST http://localhost:8000/api/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the failover strategy?"}'
```

---

## 📋 6. Current Implementation Status & Roadmap

| Subsystem / Phase | Status | Details |
| :--- | :--- | :--- |
| **Phase 1: Dual Audio & ASR** | **Completed ✅** | PipeWire loopback tap + USB mic, Silero VAD, Whisper CUDA ASR, SQLite WAL persistence, WebSocket broadcasting. |
| **Phase 2: Shorthand Engine** | **Completed ✅** | Monotonic keystroke timestamps, $[T-45\text{s}, T+15\text{s}]$ window slicing, Gemma 4 grounded minute expansion with `[ref: MM:SS]`. |
| **Phase 3: Local Knowledge RAG** | **Completed ✅** | Watchdog file observer, Markdown/PDF chunker, EmbeddingGemma 2 (768-D), SQLite VectorStore, Live Assistant dual-grounded chat. |
| **Phase 4: Ambient Desktop UX** | **In Progress 🔄** | Split-screen Web Studio operational. Upcoming: floating system tray/pill widget and audio playback seekers on citation click. |

---

## 💡 7. Important Guidelines for AI Agents

1. **When Modifying RAG Chunker:** Preserve section titles and page numbers in `DocumentChunk`; these are critical for citation badges.
2. **When Changing Database Schema:** Update `daemon/db/schema.sql` and corresponding methods in `daemon/db/repository.py`. Ensure tables use `IF NOT EXISTS`.
3. **When Extending Audio Capture:** Ensure `pw-record` process termination is clean via `SIGTERM` and queue reading tasks handle `asyncio.CancelledError`.
4. **When Handling Embeddings:** Ensure vectors remain 768-dimensional float32 arrays matching `EmbeddingGemma 2` specifications.
5. **Enforce Private Document Boundaries:** Always check `git status` before committing or advising the user on Git operations. Never stage or commit internal planning, learning, or review files (`FORESIGHT_REPORT.md`, `*PLAN*.md`, `*REVIEW*.md`, `*LEARNING*.md`, etc.). Maintain `.gitignore` rules to exclude them automatically.
