"""Test-only stubs for optional heavy dependencies.

The project imports `deepspeed` (RMU) and `lm_eval` (evaluators) at module
level; the CPU smoke tests need neither, so they are stubbed when missing. The
stubs are never imported by `fwt/` itself, only by the tests.
"""

from __future__ import annotations
import importlib.machinery
import sys
import types


def stub_deepspeed() -> None:
    if "deepspeed" in sys.modules:
        return
    module = types.ModuleType("deepspeed")

    class DeepSpeedEngine:  # minimal placeholder for isinstance() checks
        pass

    module.DeepSpeedEngine = DeepSpeedEngine  # type: ignore[attr-defined]
    module.initialize = lambda *a, **k: (None,)  # type: ignore[attr-defined]
    # accelerate inspects __spec__; without distribution metadata it still
    # concludes (correctly) that deepspeed is not really installed.
    module.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
    sys.modules["deepspeed"] = module


def stub_lm_eval() -> None:
    """Minimal `lm_eval` so that `src/evals` imports without the real package."""
    if "lm_eval" in sys.modules:
        return
    root = types.ModuleType("lm_eval")
    models = types.ModuleType("lm_eval.models")
    hf_vlms = types.ModuleType("lm_eval.models.hf_vlms")
    tasks = types.ModuleType("lm_eval.tasks")

    class HFLM:  # placeholder
        def __init__(self, *args, **kwargs):
            raise RuntimeError("lm_eval is stubbed in tests")

    class TaskManager:  # placeholder
        def __init__(self, *args, **kwargs):
            raise RuntimeError("lm_eval is stubbed in tests")

    hf_vlms.HFLM = HFLM  # type: ignore[attr-defined]
    tasks.TaskManager = TaskManager  # type: ignore[attr-defined]
    models.hf_vlms = hf_vlms  # type: ignore[attr-defined]
    root.models = models  # type: ignore[attr-defined]
    root.tasks = tasks  # type: ignore[attr-defined]
    root.simple_evaluate = lambda *a, **k: {}  # type: ignore[attr-defined]
    for name, module in (
        ("lm_eval", root), ("lm_eval.models", models),
        ("lm_eval.models.hf_vlms", hf_vlms), ("lm_eval.tasks", tasks),
    ):
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = module


def stub_optional_deps() -> None:
    """Stub every optional dependency the smoke tests do not exercise."""
    stub_deepspeed()
    stub_lm_eval()
