"""SI command-line interface.

Usage (after pip install -e .):

    si smoke          # run the smoke test (Phase 0 sanity)
    si phase1         # run Phase 1 AZR reproduction
    si phase2 <run>   # run Phase 2 multi-branch Elo
    si anchor <ckpt>  # run anchor eval on a single checkpoint
    si status         # dump current run state
"""

from __future__ import annotations

import typer
from rich import print

app = typer.Typer(add_completion=False, no_args_is_help=True, help="SI — self-improving agent stack.")


@app.command()
def smoke() -> None:
    """Phase 0 smoke: load base Gemma 4, generate a short completion, verify the toolchain."""
    print("[bold cyan]SI smoke test[/bold cyan]")
    print("Not yet implemented. See [yellow]scripts/00_smoke_test.sh[/yellow].")
    raise typer.Exit(code=1)


@app.command()
def phase1(config: str = typer.Option("configs/tier1_e4b.yaml", help="run config YAML")) -> None:
    """Phase 1: reproduce AZR on Gemma 4 E4B (single branch)."""
    print(f"[bold cyan]Phase 1 (AZR single-branch) — config {config}[/bold cyan]")
    print("Not yet implemented. See [yellow]docs/04-implementation.md §1[/yellow].")
    raise typer.Exit(code=1)


@app.command()
def phase2(config: str = typer.Option("configs/tier2_26b.yaml", help="run config YAML")) -> None:
    """Phase 2: multi-branch Elo tournament."""
    print(f"[bold cyan]Phase 2 (Elo tournament) — config {config}[/bold cyan]")
    print("Not yet implemented. See [yellow]docs/04-implementation.md §2[/yellow].")
    raise typer.Exit(code=1)


@app.command()
def anchor(
    checkpoint: str = typer.Argument(..., help="path to HF checkpoint"),
    benchmark: str = typer.Option("humaneval_plus", help="primary | secondary | meta"),
) -> None:
    """Evaluate a single checkpoint against an anchor benchmark."""
    print(f"[bold cyan]Anchor eval[/bold cyan] — ckpt={checkpoint} benchmark={benchmark}")
    print("Not yet implemented. See [yellow]docs/05-evaluation.md[/yellow].")
    raise typer.Exit(code=1)


@app.command()
def status() -> None:
    """Dump current run state (last anchor, Elo ranks, pending migrations)."""
    print("[bold cyan]SI status[/bold cyan] — not yet implemented.")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
