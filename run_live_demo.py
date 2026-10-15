#!/usr/bin/env python3
"""
Interactive Live Demo for Spark-Foresight (Phase 1).
Demonstrates real-time microphone capture, Silero VAD, Whisper ASR on GB10 CUDA,
and SQLite persistence.
"""

import asyncio
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from daemon.audio.capture import discover_audio_nodes
from daemon.asr.engine import ASREngine
from daemon.asr.timestamps import format_timestamp_ms
from daemon.db.repository import DatabaseRepository
from daemon.session_manager import LiveSession

console = Console()


async def main():
    console.print(
        Panel.fit(
            "[bold cyan]🔭 OpenForesight: Live On-Device Meeting Intelligence Demo[/bold cyan]\n"
            "[green]Platform:[/green] Linux (x86_64 / ARM64) with NVIDIA GPU & PipeWire\n"
            "[green]Audio:[/green] Native Dual-Stream Capture (Microphone + System Output Loopback)\n"
            "[green]Privacy:[/green] 100% On-Device, Zero Cloud Egress",
            border_style="cyan",
        )
    )

    # 1. Discover devices
    console.print("\n[bold yellow]🔍 Step 1: Discovering PipeWire Audio Nodes...[/bold yellow]")
    nodes = discover_audio_nodes()
    dev_table = Table(title="Detected Audio Nodes")
    dev_table.add_column("Type", style="cyan")
    dev_table.add_column("Name", style="magenta")
    dev_table.add_column("Description", style="white")

    for n in nodes:
        dev_table.add_row(n.media_class, n.name, n.description)
    console.print(dev_table)

    sources = [n for n in nodes if n.media_class == "Audio/Source"]
    sinks = [n for n in nodes if n.media_class == "Audio/Sink"]

    mic_node = sources[0].name if sources else None
    sink_node = sinks[0].name if sinks else None

    console.print(f"[green]✓ Selected Mic:[/green] {mic_node}")
    console.print(f"[green]✓ Selected Sink Monitor:[/green] {sink_node}")

    # 2. Initialize Database & ASR Engine on GPU
    console.print("\n[bold yellow]⚡ Step 2: Initializing CUDA ASR Engine...[/bold yellow]")
    repo = DatabaseRepository("foresight.db")
    await repo.initialize()

    asr = ASREngine(model_id="openai/whisper-tiny")
    console.print(f"[green]✓ ASR Engine Loaded on {asr.device}[/green]")

    # 3. Create Session
    session_id = f"demo-{int(time.time())}"
    live_session = LiveSession(
        session_id=session_id,
        title="Live Terminal Demo",
        repository=repo,
        asr_engine=asr,
        mic_node=mic_node,
        sink_node=sink_node,
    )

    # Custom subscriber to display live transcripts
    class TerminalSubscriber:
        async def send_json(self, msg: dict):
            if msg.get("type") == "transcript_segment":
                ts = msg.get("timestamp_formatted", "00:00")
                speaker = msg.get("speaker_label", "Speaker")
                channel = msg.get("speaker_channel", "mixed")
                text = msg.get("text", "")
                badge = "[cyan][Me][/cyan]" if channel == "local" else "[yellow][Remote][/yellow]"
                console.print(f"  [dim]{ts}[/dim] {badge} [bold white]{text}[/bold white]")

    sub = TerminalSubscriber()
    live_session.register_client(sub)

    console.print("\n[bold yellow]🎙️  Step 3: Starting Real-Time Audio Capture...[/bold yellow]")
    await live_session.start()

    console.print(
        Panel(
            "[bold green]Session is LIVE! Please speak into your microphone or play sound through speakers.[/bold green]\n"
            "[italic]Real-time transcripts will appear below as you speak...[/italic]\n"
            "[dim]Running for 15 seconds (press Ctrl+C anytime to stop early).[/dim]",
            border_style="green",
        )
    )

    try:
        for i in range(15):
            await asyncio.sleep(1)
            sys.stdout.write(f"\r[Recording... {15 - i}s remaining] ")
            sys.stdout.flush()
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]Stopping early on user request...[/yellow]")

    console.print("\n\n[bold yellow]⏹️  Step 4: Finalizing Session & Committing to SQLite...[/bold yellow]")
    await live_session.stop()
    console.print("[green]✓ Session stopped cleanly.[/green]")

    # 4. Review results from DB
    segments = await repo.get_transcript_segments(session_id)
    console.print(f"\n[bold green]📊 Final Session Summary ({len(segments)} segments committed to SQLite):[/bold green]")

    if segments:
        summary_table = Table(title=f"Session Segments ({session_id})")
        summary_table.add_column("Time", style="cyan")
        summary_table.add_column("Channel", style="magenta")
        summary_table.add_column("Speaker", style="yellow")
        summary_table.add_column("Transcript", style="white")

        for s in segments:
            summary_table.add_row(
                f"{format_timestamp_ms(s['start_ms'])} - {format_timestamp_ms(s['end_ms'])}",
                s["speaker_channel"],
                s["speaker_label"],
                s["text"],
            )
        console.print(summary_table)
    else:
        console.print("[dim]No speech detected during the 15-second window.[/dim]")

    console.print("\n[bold cyan]🎉 Demo completed successfully![/bold cyan]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
