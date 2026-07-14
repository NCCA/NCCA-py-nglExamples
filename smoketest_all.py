#!/usr/bin/env -S uv run --script
"""
Command-line smoke test runner for PyNGL demos.

Discovers every executable demo script the same way RunDemos.py does, then
runs each one with `--smoketest <ms>` in turn, reporting a pass/fail summary.
"""

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Default smoketest run time in milliseconds, matching the default used by
# the demos' own `--smoketest` argparse option.
DEFAULT_SMOKETEST_MS = 200

EXCLUDE_DIRS = {
    ".venv",
    ".git",
    ".worktrees",
    "__pycache__",
    ".ruff_cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ropeproject",
}


@dataclass
class Demo:
    """A data class to hold information about a single demo."""

    button_name: str
    root_path: str
    app_full_path: str


def find_executables(root: Path) -> list[Demo]:
    """
    Recursively finds all executable Python scripts to be treated as demos.

    Mirrors RunDemos.py's `_find_executables`.
    """
    exclude_stems = {Path(__file__).stem}

    def walk(current_root: Path) -> Iterator[Path]:
        for p in current_root.iterdir():
            if p.is_dir():
                if p.name in EXCLUDE_DIRS:
                    continue
                yield from walk(p)
            elif p.suffix == ".py":
                if p.stem in exclude_stems:
                    continue
                if os.access(p, os.X_OK):
                    yield p

    scripts = [p for p in walk(root) if p.stem != "RunDemos"]

    counts: dict[str, int] = {}
    for p in scripts:
        counts[p.parent.name] = counts.get(p.parent.name, 0) + 1

    executables: list[Demo] = []
    for p in scripts:
        folder_name = p.parent.name
        if counts[folder_name] > 1:
            button_name = f"{folder_name} — {p.name}"
        else:
            button_name = folder_name
        executables.append(
            Demo(
                button_name=button_name,
                root_path=str(p.parent),
                app_full_path=str(p),
            )
        )
    return executables


def run_smoketests(executables: list[Demo], smoketest_ms: int) -> list[str]:
    """
    Runs every demo with `--smoketest <ms>` in turn, returning a list of
    failure descriptions (empty if everything passed).
    """
    total = len(executables)
    failures: list[str] = []
    for index, demo in enumerate(executables, start=1):
        print(f"[{index}/{total}] {demo.button_name}...", flush=True)
        command = f"{shlex.quote(demo.app_full_path)} --smoketest {smoketest_ms}"
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=demo.root_path,
                timeout=(smoketest_ms / 1000.0) + 30.0,
            )
            if result.returncode != 0:
                failures.append(demo.button_name)
        except subprocess.TimeoutExpired:
            failures.append(f"{demo.button_name} (timed out)")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run --smoketest for every discovered PyNGL demo."
    )
    parser.add_argument(
        "--time",
        type=int,
        default=DEFAULT_SMOKETEST_MS,
        help=f"Smoketest run time in milliseconds (default: {DEFAULT_SMOKETEST_MS})",
    )
    args = parser.parse_args()

    root = Path.cwd()
    executables = find_executables(root)
    if not executables:
        print("No executable demos found.")
        return 1

    failures = run_smoketests(executables, args.time)

    total = len(executables)
    if failures:
        print(f"\nSmoketest FAILED for {len(failures)}/{total} demo(s):")
        for name in failures:
            print(f"  - {name}")
        return 1

    print(f"\nSmoketest OK for all {total} demos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
