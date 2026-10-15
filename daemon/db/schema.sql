-- SQLite Schema for Spark-Foresight (Transcripts, Sessions, Notes, and Knowledge Chunks)

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'Untitled Meeting',
    started_at INTEGER NOT NULL, -- Epoch milliseconds
    ended_at INTEGER,           -- Epoch milliseconds
    status TEXT NOT NULL CHECK(status IN ('recording', 'stopped', 'archived')) DEFAULT 'recording'
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    start_ms INTEGER NOT NULL,   -- Offset from session start in milliseconds
    end_ms INTEGER NOT NULL,     -- Offset from session start in milliseconds
    speaker_channel TEXT NOT NULL CHECK(speaker_channel IN ('local', 'remote', 'mixed')) DEFAULT 'mixed',
    speaker_label TEXT NOT NULL DEFAULT 'Speaker',
    text TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transcripts_lookup 
ON transcript_segments(session_id, start_ms, end_ms);

CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL, -- Offset from session start in milliseconds
    updated_at_ms INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    enriched_text TEXT,
    status TEXT NOT NULL CHECK(status IN ('draft', 'enriching', 'enriched')) DEFAULT 'draft',
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_notes_lookup 
ON notes(session_id, created_at_ms);

-- Knowledge Base Documents
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    modified_at INTEGER NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_path 
ON documents(file_path);

-- Knowledge Base Chunks with 768-D Dense Embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    page_number INTEGER DEFAULT 1,
    section_title TEXT DEFAULT '',
    embedding BLOB, -- 768 float32 binary values (3072 bytes)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc 
ON document_chunks(document_id);
