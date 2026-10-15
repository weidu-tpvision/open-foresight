#!/usr/bin/env python3
"""
CLI script to index local markdown & PDF documentation into Foresight Knowledge Base
using EmbeddingGemma 2 (768-D dense vectors) on local CUDA / Ollama.
"""

import asyncio
import os
import sys
import aiosqlite
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from daemon.db.repository import DatabaseRepository
from daemon.rag.vector_store import VectorStore
from daemon.rag.embeddings import EmbeddingGemmaClient
from daemon.rag.chunker import DocumentChunker
import argparse
from daemon.rag.assistant import LiveAssistant
from daemon.llm.client import GemmaClient

console = Console()


async def main():
    parser = argparse.ArgumentParser(
        description="Index local Markdown and PDF documentation into Foresight Knowledge Base."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=str(Path.cwd().resolve()),
        help="Target workspace or directory to index (default: current directory)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Purge all existing indexed documents before indexing new ones",
    )
    args = parser.parse_args()

    target_path = Path(args.target).resolve()

    console.print(
        f"[bold cyan]📁 Foresight Knowledge Ingestion[/bold cyan]\n"
        f"Target Workspace: [green]{target_path}[/green]\n"
        f"Embedding Model: [purple]EmbeddingGemma 2 (768-D)[/purple]\n"
        f"LLM Assistant: [purple]Gemma 4 (Local CUDA / Ollama)[/purple]\n"
    )

    if not target_path.exists():
        console.print(f"[red]Error: Path {target_path} does not exist.[/red]")
        sys.exit(1)

    repo = DatabaseRepository("foresight.db")
    await repo.initialize()

    # Optional: Clean existing knowledge base if explicitly requested
    if args.clean:
        console.print("[yellow]🧹 Purging existing knowledge base documents (--clean)...[/yellow]")
        async with aiosqlite.connect("foresight.db") as db:
            await db.execute("DELETE FROM document_chunks")
            await db.execute("DELETE FROM documents")
            await db.commit()
        console.print("[green]✓ Clean knowledge base ready (all previous documents purged).[/green]\n")

    # 2. Discover high-value markdown docs across workspace
    console.print("[yellow]🔍 Scanning workspace for high-signal documentation...[/yellow]")
    doc_files: list[Path] = []
    skip_dirs = {
        ".git", ".venv", "venv", "node_modules", "build", "dist",
        "__pycache__", "target", ".cache", "third_party", "deps",
        "submodules", "external", "include", ".idea", ".vscode"
    }

    for root, dirs, files in os.walk(target_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in skip_dirs]
        for f in files:
            if f.endswith(".md") and not f.startswith("."):
                # Skip secondary translations
                if any(f.startswith(f"README_{lang}.md") for lang in ["ar", "de", "es", "fr", "id", "it", "ja", "ko", "pl", "pt-BR", "ru", "th", "tr", "uk", "vi", "zh"]):
                    continue
                # Skip license/changelog noise
                if any(k in f.upper() for k in ["CHANGELOG", "THIRD_PARTY", "LICENSE"]):
                    continue
                file_full = Path(root) / f
                if file_full.stat().st_size > 200_000:
                    continue
                doc_files.append(file_full)

    console.print(f"[green]Found [bold]{len(doc_files)}[/bold] high-value project documents across workspace.[/green]\n")

    # 3. Initialize VectorStore with optimized 1200-char semantic chunks
    chunker = DocumentChunker(target_chunk_size=1200, chunk_overlap=120)
    embed_client = EmbeddingGemmaClient()
    vs = VectorStore("foresight.db", embedding_client=embed_client, chunker=chunker)

    # 4. Index documents with real-time progress
    total_chunks = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Embedding with EmbeddingGemma 2...", total=len(doc_files))

        for doc_file in doc_files:
            rel_name = doc_file.relative_to(target_path)
            progress.update(task, description=f"[cyan]Indexing [bold white]{rel_name}[/bold white]...")
            try:
                res = await vs.index_file(doc_file)
                total_chunks += res["chunks_indexed"]
            except Exception as e:
                console.print(f"[red]Failed {rel_name}: {e}[/red]")
            progress.advance(task)

    console.print(
        f"\n[bold green]✓ Ingestion Complete![/bold green] Indexed [bold cyan]{len(doc_files)}[/bold cyan] documents "
        f"into [bold cyan]{total_chunks}[/bold cyan] 768-D semantic chunks in SQLite.\n"
    )

    # 5. Verification Test Queries
    console.print("[bold yellow]🧪 Running Sample Verification Queries with Live Assistant...[/bold yellow]")
    gemma = GemmaClient()
    assistant = LiveAssistant(vector_store=vs, repository=repo, llm_client=gemma)

    test_queries = [
        "What are the main engineering projects, architecture, and goals described in these documents?",
        "What are the key technical workflows, components, and implementation details?",
    ]

    for q in test_queries:
        console.print(f"\n[bold white]Query:[/bold white] \"{q}\"")
        resp = await assistant.answer_query(q)
        console.print(f"\n[bold green]Assistant Answer (Gemma 4):[/bold green]\n{resp.answer_markdown}\n")

        if resp.citations:
            cit_table = Table(title="Retrieved Citations (EmbeddingGemma 2 768-D Cosine Search)")
            cit_table.add_column("Type", style="cyan")
            cit_table.add_column("Source", style="magenta")
            cit_table.add_column("Score", style="green")

            for c in resp.citations:
                cit_table.add_row(c.source_type, c.source_label, f"{c.score:.4f}")
            console.print(cit_table)


if __name__ == "__main__":
    asyncio.run(main())
