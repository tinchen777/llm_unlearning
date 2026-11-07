"""Cost of a method: wall-clock seconds and peak GPU memory of its stages (the evaluation is NOT included), written next to the results.

    meter = CostMeter()
    with meter.stage("record"):
        ...
    meter.save(path)        # {"stages": {"record": {"seconds": .., "peak_allocated_gb": .., "peak_reserved_gb": ..}}, "total_seconds": .., "peak_allocated_gb": ..}

The peak is `torch.cuda.max_memory_allocated` since the start of the stage (reset at every stage), so it is the model + the activations / optimizer state of this process,
not what `nvidia-smi` shows (the caching allocator keeps more reserved). Without CUDA the memory is `None`.
"""
from __future__ import annotations
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import torch

GB = 2 ** 30


class CostMeter:
    def __init__(self) -> None:
        self.stages: Dict[str, Dict[str, Optional[float]]] = {}
        self.extra: Dict[str, Any] = {}

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextmanager
    def stage(self, name: str):
        self._sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            cuda = torch.cuda.is_available()
            done = {
                "seconds": time.perf_counter() - start,
                "peak_allocated_gb": torch.cuda.max_memory_allocated() / GB if cuda else None,
                "peak_reserved_gb": torch.cuda.max_memory_reserved() / GB if cuda else None,
            }
            old = self.stages.get(name)  # a stage that is entered several times adds up (time) / takes the maximum (memory)
            if old is not None:
                done["seconds"] += old["seconds"]
                for key in ("peak_allocated_gb", "peak_reserved_gb"):
                    if done[key] is not None and old[key] is not None:
                        done[key] = max(done[key], old[key])
            self.stages[name] = done

    def summary(self) -> Dict[str, Any]:
        peaks = [s["peak_allocated_gb"] for s in self.stages.values() if s["peak_allocated_gb"] is not None]
        return {
            "stages": self.stages,
            "total_seconds": sum(s["seconds"] for s in self.stages.values()),
            "peak_allocated_gb": max(peaks) if peaks else None,
            **self.extra,
        }

    def save(self, path: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True), encoding="utf-8")
