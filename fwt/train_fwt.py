#!/usr/bin/env python
"""Training entry point for Forgetting Without Telling.

Thin wrapper around `src/train.py`: it registers the FWT trainer/dataset and the
`fwt/configs` search path, then hands over to the project's own pipeline, so
every existing flag, evaluator and logger keeps working.

    python fwt/train_fwt.py --config-name=unlearn \
        experiment=unlearn/tofu/fwt \
        model=Llama-3.2-1B-Instruct \
        trainer.method_args.layers="[7]" \
        data.forget.TOFU_QA_forget_neighbor.args.neighbors_path=data/fwt/neighbors_forget10.json \
        task_name=fwt/redirect_forget10
"""

from __future__ import annotations

from fwt import _paths  # noqa: F401  (sets sys.path for `src/` imports)
from fwt.hydra_plugin import register_fwt_configs

register_fwt_configs()

import fwt.register  # noqa: E402, F401  (fills the dataset/trainer registries)

from train import main  # noqa: E402

if __name__ == "__main__":
    main()
