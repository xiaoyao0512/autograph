"""Utilities."""

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union

__all__ = [
    "git_describe",
]


def git_describe(path: Path) -> Optional[str]:
    """Obtain git describe output for path."""
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty", "--tags", "--long"],
            check=True,
            text=True,
            capture_output=True,
            cwd=path,
        ).stdout.splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def slurm_info() -> Union[Tuple[int, int], Tuple[None, None]]:
    """SLURM process information."""
    process_id_var = os.environ.get("SLURM_PROCID")
    processes_var = os.environ.get("SLURM_STEP_NUM_TASKS")
    if process_id_var is None or processes_var is None:
        process_id = None
        processes = None
        result: Union[Tuple[int, int], Tuple[None, None]] = (
            process_id,
            processes,
        )
    else:
        process_id = int(process_id_var)
        processes = int(processes_var)
        result = process_id, processes
    return result
