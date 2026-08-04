"""Lightweight metrics tracker for JEPA-WAM training."""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


class Metrics:
    """Simple metrics accumulator with JSONL logging."""

    def __init__(self, run_dir: Path, resume_step: int = 0) -> None:
        self.run_dir = run_dir
        self.global_step = resume_step
        self.start_time = time.time()
        self.history = []

        self.jsonl_path = run_dir / "metrics.jsonl"
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def commit(self, **kwargs) -> None:
        """Log a dictionary of scalar metrics for the current step."""
        entry = {"step": self.global_step, **kwargs}
        self.history.append(entry)

        # Write to JSONL
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def step(self) -> None:
        self.global_step += 1

    def get_status(self) -> str:
        if not self.history:
            return "Starting..."
        latest = self.history[-1]
        parts = [f"{k}={v:.4f}" for k, v in latest.items() if isinstance(v, (int, float)) and k != "step"]
        return f"step={latest['step']} " + " ".join(parts)

    def finalize(self) -> None:
        pass
