"""
HuggingFace Qwen-VL wrapper with spatial-context fusion interface.

The ``set_spatial_context`` method stores a 3-D shape embedding, a
natural-language shape profile, and the estimated Z coordinate so that
the next ``query`` call can incorporate geometric awareness.

The actual model loading and inference are stubs — replace
``load_model`` and ``query`` with real HF Transformers code when ready.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class VLMAgent:
    def __init__(self, model_name: str = "Qwen/Qwen-VL"):
        self.model_name = model_name
        self.model = None

        # Spatial context (set before each query)
        self._embedding: Optional[np.ndarray] = None
        self._shape_text: Optional[str] = None
        self._z_est: Optional[float] = None

    # ── Spatial context interface ─────────────────────────────────────────────
    def set_spatial_context(
        self,
        embedding: np.ndarray,
        shape_text: str,
        z_est: float,
    ) -> None:
        """
        Store 3-D spatial context for the next ``query`` call.

        Parameters
        ----------
        embedding   : 1-D array (e.g. 256-d PointNet global feature).
        shape_text  : Natural-language shape profile from ShapeDescriptor.
        z_est       : Robust depth estimate in metres.
        """
        self._embedding = embedding
        self._shape_text = shape_text
        self._z_est = z_est

    def has_spatial_context(self) -> bool:
        return self._embedding is not None

    # ── Model lifecycle ───────────────────────────────────────────────────────
    def load_model(self):
        raise NotImplementedError(
            "VLMAgent.load_model() — plug in HF Transformers loading here"
        )

    def query(self, image, prompt: str) -> str:
        """
        Run VLM inference.

        When spatial context is available the prompt is augmented with the
        shape profile text and Z estimate.  The embedding vector would be
        fused into the model's hidden state (implementation TBD).
        """
        raise NotImplementedError(
            "VLMAgent.query() — plug in model.generate() here"
        )
