"""CLI entrypoint and command runner for OpenForesight (Spark-Foresight)."""

from __future__ import annotations

import argparse
import sys
import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="open-foresight",
        description="100% On-Device Meeting & Knowledge Intelligence for Linux (NVIDIA / PipeWire / Gemma 4 / Ollama)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI backend daemon and Web Studio")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")

    # Command: index
    index_parser = subparsers.add_parser("index", help="Index Markdown and PDF files into the local vector store")
    index_parser.add_argument("target", nargs="?", default=".", help="Directory or file path to index (default: .)")
    index_parser.add_argument("--clean", action="store_true", help="Purge existing knowledge documents before indexing")

    # Command: demo
    subparsers.add_parser("demo", help="Run the interactive terminal live microphone capture demo")

    args = parser.parse_args()

    if args.command == "serve" or args.command is None:
        if args.command is None:
            # If no subcommand passed, default to serve
            host = "0.0.0.0"
            port = 8000
            reload = False
        else:
            host = args.host
            port = args.port
            reload = args.reload

        print(f"🔭 Starting OpenForesight Daemon on http://{host}:{port} ...")
        uvicorn.run("daemon.app:app", host=host, port=port, reload=reload)

    elif args.command == "index":
        import asyncio
        from index_workspace import main as index_main
        # Override sys.argv for index_workspace argument parser
        sys.argv = ["index_workspace.py", args.target]
        if args.clean:
            sys.argv.append("--clean")
        asyncio.run(index_main())

    elif args.command == "demo":
        import asyncio
        from run_live_demo import main as demo_main
        asyncio.run(demo_main())

    else:
        parser.print_help()
