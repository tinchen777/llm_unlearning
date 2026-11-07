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
    """Stages may be nested (an inner stage inside an outer one): the outer stage's `seconds` then EXCLUDES the inner ones (so the sum over the stages is the real total) and
    its peak memory is the maximum over its own part and the inner stages."""

    def __init__(self) -> None:
        self.stages: Dict[str, Dict[str, Optional[float]]] = {}
        self.extra: Dict[str, Any] = {}
        self._stack: list = []

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _peaks() -> tuple:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / GB, torch.cuda.max_memory_reserved() / GB
        return None, None

    @contextmanager
    def stage(self, name: str):
        self._sync()
        if self._stack:  # keep what the enclosing stage has seen so far (the counter is reset for this stage)
            self._stack[-1]["peaks"] = self._merge(self._stack[-1]["peaks"], self._peaks())
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        frame = {"child_seconds": 0.0, "peaks": (None, None)}
        self._stack.append(frame)
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            self._stack.pop()
            peaks = self._merge(frame["peaks"], self._peaks())
            if self._stack:
                parent = self._stack[-1]
                parent["child_seconds"] += elapsed
                parent["peaks"] = self._merge(parent["peaks"], peaks)
            done = {"seconds": elapsed - frame["child_seconds"], "peak_allocated_gb": peaks[0], "peak_reserved_gb": peaks[1]}
            old = self.stages.get(name)  # a stage that is entered several times adds up (time) / takes the maximum (memory)
            if old is not None:
                done["seconds"] += old["seconds"]
                for key in ("peak_allocated_gb", "peak_reserved_gb"):
                    done[key] = self._max(done[key], old[key])
            self.stages[name] = done

    @staticmethod
    def _max(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return b if a is None else a if b is None else max(a, b)

    def _merge(self, a: tuple, b: tuple) -> tuple:
        return self._max(a[0], b[0]), self._max(a[1], b[1])

    def summary(self) -> Dict[str, Any]:
        peaks = [s["peak_allocated_gb"] for s in self.stages.values() if s["peak_allocated_gb"] is not None]
        return {
            "stages": self.stages,
            "total_seconds": sum(s["seconds"] for s in self.stages.values()),
            "peak_allocated_gb": max(peaks) if peaks else None,
            "nested_stages_exclusive": True,
            **self.extra,
        }

    def save(self, path: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True), encoding="utf-8")
