"""Register the FWT dataset and trainer in the project's registries.

Importing this module is the only wiring needed - `src/` is left untouched:

```python
import fwt.register  # noqa: F401
```

`fwt/train_fwt.py` does this for you.
"""

from __future__ import annotations
import logging

from fwt import _paths  # noqa: F401  (sets sys.path)

from data import DATASET_REGISTRY
from trainer import TRAINER_REGISTRY

from fwt.data import NeighborRedirectDataset
from fwt.trainer import NeighborRedirect

logger = logging.getLogger(__name__)

DATASET_REGISTRY[NeighborRedirectDataset.__name__] = NeighborRedirectDataset
TRAINER_REGISTRY[NeighborRedirect.__name__] = NeighborRedirect

logger.debug(
    "Registered FWT components: %s (dataset), %s (trainer)",
    NeighborRedirectDataset.__name__, NeighborRedirect.__name__,
)

__all__ = ["NeighborRedirectDataset", "NeighborRedirect"]
