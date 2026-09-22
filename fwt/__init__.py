"""Forgetting Without Telling (FWT).

Making unlearned entities indistinguishable from strangers, by redirecting the
activations of a forgotten entity onto those of an attribute-matched synthetic
neighbour.

Three stages, each usable on its own:

1. `fwt.neighbors` - build the attribute-matched neighbour bank (`neighbors.json`).
2. `fwt.layers`    - choose the layer(s) to edit (`layer_selection.json`).
3. `fwt.trainer`   - train the redirection (`NeighborRedirect`).

Sub-packages are imported explicitly so that stage 1 stays usable without torch.
"""

from __future__ import annotations

__all__ = ["neighbors", "layers", "data", "trainer"]
__version__ = "0.1.0"
